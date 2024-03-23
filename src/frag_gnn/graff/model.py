import torch as th
import torch.nn as nn
import numpy as np
import torch_geometric as pyg

from frag_gnn.graff.nn_utils import *

class GrAFFModel(nn.Module):
	def __init__(
		self,
		vocab_size,
		num_isotope_types,
		encoder_dim,
		decoder_dim,
		encoder_depth,
		decoder_depth,
		num_eigs,
		eig_dim,
		eig_depth,
		dropout,
		min_probability,
		min_mz,
		isotopes,
		node_feats,
		edge_feats,
		**kwargs
	):
		super().__init__()

		self.encoder_dim = encoder_dim
		self.decoder_dim = decoder_dim
		self.encoder_depth = encoder_depth
		self.decoder_depth = decoder_depth
		self.num_eigs = num_eigs
		self.eig_dim = eig_dim
		self.eig_depth = eig_depth
		self.dropout = dropout

		self.vocab_size = vocab_size
		self.num_isotope_types = num_isotope_types
		# TODO: add collision energy
		self.covariates_dim = 1
		
		# inference time only
		self.min_probability = min_probability
		self.min_mz = min_mz
		
		# embedding layers
		self.onehot = CanonicalOneHot(
			node_feats=node_feats, 
			edge_feats=edge_feats)
		self.node_emb = nn.Sequential(
			nn.Linear(self.onehot.node_dim, encoder_dim),
			nn.SiLU(inplace=True),
			nn.Dropout(dropout),
			nn.LayerNorm(encoder_dim),
			nn.Linear(encoder_dim, encoder_dim)
		)
		self.edge_emb = nn.Sequential(
			nn.Linear(self.onehot.edge_dim, encoder_dim),
			nn.SiLU(inplace=True),
			nn.Dropout(dropout),
			nn.LayerNorm(encoder_dim),
			nn.Linear(encoder_dim, encoder_dim)
		)
		
		if num_eigs > 0:
			self.signnet = SignNet(
				num_eigs=num_eigs,
				embed_dim=encoder_dim,
				rho_dim=encoder_dim,
				rho_depth=eig_depth,
				phi_dim=eig_dim,
				phi_depth=eig_depth,
				dropout=dropout
			)
		else:
			self.signnet = None
		
		self.encoder = GINE(
			node_dim=encoder_dim,
			edge_dim=encoder_dim,
			model_dim=encoder_dim, 
			model_depth=encoder_depth,
			dropout=dropout
		)
		
		self.cov_emb = nn.Sequential(
			nn.Linear(self.covariates_dim, encoder_dim),
			nn.SiLU(inplace=True),
			nn.Dropout(dropout),
			nn.LayerNorm(encoder_dim),
			nn.Linear(encoder_dim, encoder_dim)
		)
		
		self.attn = nn.Linear(encoder_dim, 1)
		
		layers = []
		layers.append(nn.Linear(encoder_dim, decoder_dim))
		for _ in range(decoder_depth):
			layers.append(ResBlock(decoder_dim, dropout))
		self.decoder = nn.Sequential(*layers)
		
		self.isotope_shift = nn.Linear(decoder_dim, self.num_isotope_types)
		self.clf = nn.Linear(decoder_dim, self.vocab_size + 1)

		self.log_2 = float(np.log(2))
		self.isotopes = isotopes

	# def set_ann_library(self, library_mzs, library_loss_mask):

	# 	self.library_mzs = nn.Parameter(library_mzs.copy())
	# 	self.library_mzs.requires_grad = False
	# 	self.library_loss_mask = nn.Parameter(library_loss_mask.copy())
	# 	self.library_loss_mask.requires_grad = False

	def forward(
			self,
			g: pyg.data.Data,
			library_mzs: th.Tensor,
			isotope_mask: th.Tensor,
			dup_mask: th.Tensor,
			**kwargs):

		batch_size = len(g.ptr) - 1
		device = g.x.device
		
		# embed node, edge, eigenfeatures
		x_atom, x_bond = self.onehot(g.x, g.edge_attr)
		x_atom = self.node_emb(x_atom)
		x_bond = self.edge_emb(x_bond)
		if self.num_eigs > 0:
			x_eig = self.signnet(g.eigvecs[:,:self.num_eigs],
								 g.eigvals[:,:self.num_eigs][g.batch])
		else:
			x_eig = 0

		# run message passing
		x_mol = self.encoder(g, x_atom + x_eig, x_bond)
		# and attention-pool across atoms
		w = pyg.utils.softmax(self.attn(x_mol), g.batch)
		z = pyg.nn.global_add_pool(x_mol * w, g.batch)
		
		# condition molecule representation on covariates
		covariates = isotope_mask.reshape(batch_size,1)
		z = z + self.cov_emb(covariates)
		# transform to spectrum representation
		z = self.decoder(z)
		# and predict logits
		log_y_pred = self.clf(z)
		
		# add the (approximated) isotopic envelope
		log_y_pred = log_y_pred.view(batch_size, self.vocab_size, 1)
		isotope_shift = self.isotope_shift(z).view(batch_size, 1, self.num_isotope_types)
		log_y_pred = log_y_pred + isotope_shift
		
		# correct intensities of double-counted formulas
		log_y_pred = log_y_pred - dup_mask * self.log_2
		
		pred_mzs = library_mzs.reshape(batch_size,-1)
		pred_logprobs = log_y_pred.reshape(batch_size,-1)
		pred_batch_idxs = th.arange(batch_size,device=device).view(-1,1).repeat(1,pred_mzs.shape[1])

		out_d = {
			"pred_mzs": pred_mzs,
			"pred_logprobs": pred_logprobs,
			"pred_batch_idxs": pred_batch_idxs
		}

		return out_d


class SignNet(nn.Module):
	def __init__(self,
				 num_eigs, embed_dim,
				 phi_dim, phi_depth,
				 rho_dim, rho_depth,
				 dropout=0):
		super().__init__()
		
		self.embed_dim = embed_dim
		self.num_eigs = num_eigs
		self.phi_dim = phi_dim
		self.phi_depth = phi_depth
		self.rho_dim = rho_dim
		self.rho_depth = rho_depth
		self.dropout = dropout
		
		layers = []
		layers.append(nn.Linear(2, phi_dim))
		for _ in range(phi_depth):
			layers.extend([
				nn.Linear(phi_dim, phi_dim),
				nn.SiLU(inplace=True),
				nn.Dropout(dropout),
				nn.LayerNorm(phi_dim)
			])
		self.phi = nn.Sequential(*layers)
		
		layers = []
		if phi_dim != rho_dim:
			layers.append(nn.Linear(phi_dim, rho_dim))
		for _ in range(rho_depth):
			layers.extend([
				nn.Linear(rho_dim, rho_dim),
				nn.SiLU(inplace=True),
				nn.Dropout(dropout),
				nn.LayerNorm(rho_dim)
			])
		if embed_dim != rho_dim:
			layers.append(nn.Linear(rho_dim, embed_dim))
		self.rho = nn.Sequential(*layers)
	
	def forward(self, v, l):
		x = self.phi(th.stack([ v,l],-1)) + \
			self.phi(th.stack([-v,l],-1))
		x = self.rho(x.sum(1))
		return x