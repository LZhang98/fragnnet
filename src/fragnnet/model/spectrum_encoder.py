import torch.nn as nn

from fragnnet.model.nn_blocks import (
    GNN,
    LowRankDense,
    MLPBlocks,
    SetTransformer,
    build_pool_module,
)


class SpectrumEncoder(nn.Module):
    """
    Spectrum Encoder for processing spectrum data.

    Args:
        input_dim (int): Dimension of the input spectrum features.
        hidden_dim (int): Dimension of the hidden layers.
        output_dim (int): Dimension of the output embedding.
        num_heads (int): Number of attention heads.
        num_inds (int): Number of inducing points.
        num_blocks (int): Number of Set Transformer blocks.
        dropout (float): Dropout rate for regularization.
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
        # GNN parameters
        gnn_type="GINE",
        edge_feats_size=0,
        pool_type="mean",
    ):
        super(SpectrumEncoder, self).__init__()
        self.spectrum_encoder_type = spectrum_encoder_type

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
        elif spectrum_encoder_type == "gnn":
            self.model = GNN(
                hidden_size=hidden_dim,
                num_layers=num_blocks,
                node_feats_size=input_dim,
                edge_feats_size=edge_feats_size,
                gnn_type=gnn_type,
                dropout=dropout,
                normalization=normalization,
            )
            self.pool = build_pool_module(pool_type, hidden_dim)
            self.gnn_proj = nn.Linear(hidden_dim, output_dim)
        else:
            raise ValueError(f"Unsupported spectrum_encoder_type: {spectrum_encoder_type}")

    def forward(self, x, batch_idx=None):
        """
        Forward pass for the SetTransformer Spectrum Encoder.

        Args:
            x (Tensor): Input tensor of shape (batch_size, set_size, input_dim) or (total_elements, input_dim).
            batch_idx (Tensor, optional): Batch indices for variable-sized sets.

        Returns:
            Tensor: Encoded spectrum of shape (batch_size, output_dim).
        """
        if self.spectrum_encoder_type == "gnn":
            # x is expected to be a pyg Batch object
            batch = x
            spec_embed = self.model(batch.x, batch.batch, batch.edge_index, batch.edge_attr)
            spec_embed = self.pool(spec_embed, batch.batch)
            spec_embed = self.gnn_proj(spec_embed)
            batch_size = spec_embed.shape[0]
        else:
            # Get spectrum embedding from SetTransformer
            spec_embed = self.model(x, batch_idx=batch_idx)
            batch_size = spec_embed.shape[0]

        return spec_embed
