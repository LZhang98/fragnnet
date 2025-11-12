import torch as th
import torch.nn as nn
from fragnnet.model.nn_blocks import MLPBlocks, SetTransformer, LowRankDense
from fragnnet.model.base_model import CEModel, PrecModel, InstModel

class SpectrumEncoder(nn.Module, CEModel, PrecModel, InstModel):
	"""
	Spectrum Encoder for processing spectrum data.
	Supports combining collision energy, precursor type, and instrument type embeddings.

	Args:
		input_dim (int): Dimension of the input spectrum features.
		hidden_dim (int): Dimension of the hidden layers.
		output_dim (int): Dimension of the output embedding.
		num_heads (int): Number of attention heads.
		num_inds (int): Number of inducing points.
		num_blocks (int): Number of Set Transformer blocks.
		dropout (float): Dropout rate for regularization.
		int_embedder (str): Type of integer embedder for CE.
		ce_insert_location (str): Where to insert CE embedding ("none", "mol", "mlp").
		ce_insert_type (str): Type of CE embedding ("id", "lin", "embed", "bin").
		ce_insert_merge (bool): Whether to merge CE embeddings across batch.
		ce_insert_size (int): CE embedding size.
		ce_mean (float): Mean for CE normalization.
		ce_std (float): Std for CE normalization.
		ce_max (float): Max value for CE.
		prec_insert_location (str): Where to insert precursor embedding ("none", "mol", "mlp").
		prec_insert_size (int): Precursor embedding size.
		prec_num_types (int): Number of precursor types.
		inst_insert_location (str): Where to insert instrument embedding ("none", "mol", "mlp").
		inst_insert_size (int): Instrument embedding size.
		inst_num_types (int): Number of instrument types.
	"""
	def __init__(
		self,
		spectrum_encoder_type: str,
		input_dim,
		hidden_dim,
		output_dim,
		num_heads=4,
		num_inds=32,
		num_blocks=2,
		dropout=0.1,
		normalization="batch",
		# CE parameters
		int_embedder="one_hot",
		ce_insert_location="none",
		ce_insert_type="lin",
		ce_insert_merge=False,
		ce_insert_size=32,
		ce_mean=0.0,
		ce_std=1.0,
		ce_max=100.0,
		# Prec parameters
		prec_insert_location="none",
		prec_insert_size=32,
		prec_num_types=10,
		# Inst parameters
		inst_insert_location="none",
		inst_insert_size=32,
		inst_num_types=10,
	):
		super(SpectrumEncoder, self).__init__()

		# Initialize CE, Prec, Inst embeddings
		self._ce_init(
			int_embedder=int_embedder,
			ce_insert_location=ce_insert_location,
			ce_insert_type=ce_insert_type,
			ce_insert_merge=ce_insert_merge,
			ce_insert_size=ce_insert_size,
			ce_mean=ce_mean,
			ce_std=ce_std,
			ce_max=ce_max,
		)
		self._prec_init(
			prec_insert_location=prec_insert_location,
			prec_insert_size=prec_insert_size,
			prec_num_types=prec_num_types,
		)
		self._inst_init(
			inst_insert_location=inst_insert_location,
			inst_insert_size=inst_insert_size,
			inst_num_types=inst_num_types,
		)

		# Calculate total input dimension for spectrum encoder
		# (output of SetTransformer + optional CE/Prec/Inst embeddings)
		output_dim = output_dim
		combine_input_dim = output_dim + self.ce_mlp_input_dim + self.prec_mlp_input_dim + self.inst_mlp_input_dim

		if spectrum_encoder_type == "set_transformer":
			self.model = SetTransformer(
				input_dim=input_dim,
				hidden_dim=hidden_dim,
				output_dim=output_dim,
				num_heads=num_heads,
				num_inds=num_inds,
				num_blocks=num_blocks,
				dropout=dropout,
			)
		elif spectrum_encoder_type == "mlp":
			self.model = MLPBlocks(
				input_size=input_dim,
				hidden_size=hidden_dim,
				normalization=normalization,
				dropout=dropout,
				num_layers=num_blocks,
				output_size=output_dim,
				use_residuals=True,
			)
		elif spectrum_encoder_type == "low_rank_dense":
			# LowRankDense is a single layer, so we stack multiple with dropout
			rank = hidden_dim // 2  # Use half of hidden_dim as rank for compression
			layers = []
			for i in range(num_blocks):
				in_dim = input_dim if i == 0 else hidden_dim
				out_dim = output_dim if i == num_blocks - 1 else hidden_dim
				layers.append(LowRankDense(in_dim, out_dim, rank))
				if i < num_blocks - 1:  # No activation/dropout on final layer
					layers.append(nn.ReLU())
					layers.append(nn.Dropout(dropout))
			self.model = nn.Sequential(*layers)
		else:
			raise ValueError(f"Unsupported spectrum_encoder_type: {spectrum_encoder_type}")
		# Final projection if we're combining embeddings
		if combine_input_dim > output_dim:
			self.combine_projection = nn.Linear(combine_input_dim, output_dim)
		else:
			self.combine_projection = nn.Identity()

	def _ce_location_check(self):
		"""Check that CE location is valid for spectrum encoder."""
		assert self.ce_insert_location in ["none", "mlp"], \
			f"SetTransformerSpectrumEncoder only supports ce_insert_location='none' or 'mlp', got '{self.ce_insert_location}'"

	def _prec_location_check(self):
		"""Check that Prec location is valid for spectrum encoder."""
		assert self.prec_insert_location in ["none", "mlp"], \
			f"SetTransformerSpectrumEncoder only supports prec_insert_location='none' or 'mlp', got '{self.prec_insert_location}'"

	def _inst_location_check(self):
		"""Check that Inst location is valid for spectrum encoder."""
		assert self.inst_insert_location in ["none", "mlp"], \
			f"SetTransformerSpectrumEncoder only supports inst_insert_location='none' or 'mlp', got '{self.inst_insert_location}'"

	def forward(self, x, batch_idx=None, ce=None, ce_batch_idxs=None, prec_type=None, inst_type=None):
		"""
		Forward pass for the SetTransformer Spectrum Encoder.

		Args:
			x (Tensor): Input tensor of shape (batch_size, set_size, input_dim) or (total_elements, input_dim).
			batch_idx (Tensor, optional): Batch indices for variable-sized sets.
			ce (Tensor, optional): Collision energy values.
			ce_batch_idxs (Tensor, optional): Batch indices for CE values.
			prec_type (Tensor, optional): Precursor type indices.
			inst_type (Tensor, optional): Instrument type indices.

		Returns:
			Tensor: Encoded spectrum of shape (batch_size, output_dim).
		"""
		# Get spectrum embedding from SetTransformer
		spec_embed = self.model(x, batch_idx=batch_idx)
		batch_size = spec_embed.shape[0]

		# Collect embeddings to combine
		embeddings = [spec_embed]

		# Add CE embedding if configured
		if self.ce_insert_location == "mlp" and ce is not None:
			if ce_batch_idxs is None:
				ce_batch_idxs = th.arange(batch_size, device=ce.device)
			ce_embed = self.embed_ce(ce, ce_batch_idxs, batch_size)
			embeddings.append(ce_embed)

		# Add Prec embedding if configured
		if self.prec_insert_location == "mlp" and prec_type is not None:
			prec_embed = self.embed_prec(prec_type)
			embeddings.append(prec_embed)

		# Add Inst embedding if configured
		if self.inst_insert_location == "mlp" and inst_type is not None:
			inst_embed = self.embed_inst(inst_type)
			embeddings.append(inst_embed)

		# Concatenate all embeddings
		if len(embeddings) > 1:
			combined = th.cat(embeddings, dim=-1)
			output = self.combine_projection(combined)
		else:
			output = spec_embed

		return output