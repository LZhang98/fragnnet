"""
Siamese  Model for MCES regression pretraining
The model uses GNN  and optional embeddings for collision energy (CE),
precursor type, and instrument type, and predicts spectra using a feedforward neural network.
"""

import torch as th
import torch.nn as nn
import torch_geometric as pyg
from fragnnet.model.base_model import CEModel, PrecModel, InstModel
from fragnnet.utils.feat_utils import get_mol_feats_sizes
from fragnnet.model.nn_blocks import MLPBlocks, GNN, build_pool_module
class SiameseGNNModel(nn.Module, CEModel, PrecModel, InstModel):
    """
    Siamese GNN model for molecular graph embedding and MCES regression.
    Uses real CE, Prec, and Inst embedders (like FragGNNModel), but use default values in forward.
    This is done to maintain compatibility and without changing the data pipeline.
    """
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

        # how to combine two mol embeddings before regression. Supported:
        #  - 'c' (default): absolute element-wise difference |xi - xj| (order-invariant)
        #  - 'signed_diff': signed element-wise difference xi - xj (order-sensitive)
        #  - 'concat_avg': concatenate [xi, xj] and average predictions for both orders
        #  - 'cross_attn': cross-attention between the two embeddings (symmetric by averaging both directions)
        combination_mode: str = "cross_attn",
        # cross-attention hyperparameters (only used when combination_mode=='cross_attn')
        cross_attn_dim: int | None = None,
        cross_attn_heads: int = 4,
    ):
        super().__init__()
        self.combination_mode = combination_mode
        self.cross_attn_dim = cross_attn_dim
        self.cross_attn_heads = cross_attn_heads
        # Real CE, Prec, Inst embedders (same as FragGNNModel)
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
        self._prec_init(
            prec_insert_location=prec_insert_location,
            prec_insert_size=prec_insert_size,
            prec_num_types=len(prec_types)
        )
        self._inst_init(
            inst_insert_location=inst_insert_location,
            inst_insert_size=inst_insert_size,
            inst_num_types=len(inst_types)
        )

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
        self.mol_pool = build_pool_module(mol_pool_type, mol_hidden_size)

        # MLP input = GNN output
        self.mlp_input_dim = mol_hidden_size
        self.mlp_input_dim += self.ce_mlp_input_dim + self.prec_mlp_input_dim + self.inst_mlp_input_dim

        # Determine regressor input size based on combination mode.
        # For concat modes we need double the input dimension.
        if self.combination_mode in ("concat_avg", "concat"):
            reg_input_size = 2 * self.mlp_input_dim
        elif self.combination_mode in ("abs_diff", "signed_diff"):
            reg_input_size = self.mlp_input_dim
        elif self.combination_mode == "cross_attn":
            # determine cross attention projection dimension
            if self.cross_attn_dim is None:
                # default to mlp input dim
                self.cross_attn_dim = self.mlp_input_dim
            # ensure number of heads divides embed dim
            heads = min(self.cross_attn_heads, max(1, self.cross_attn_dim))
            while heads > 1 and (self.cross_attn_dim % heads) != 0:
                heads -= 1
            self._cross_attn_heads = heads
            # register small projection + multihead attention for cross interaction
            self.cross_proj = nn.Linear(self.mlp_input_dim, self.cross_attn_dim)
            self.cross_mha = nn.MultiheadAttention(embed_dim=self.cross_attn_dim, num_heads=self._cross_attn_heads, batch_first=True)
            # combined vector will contain [xi, attn_ij, xj, attn_ji]
            reg_input_size = 2 * self.mlp_input_dim + 2 * self.cross_attn_dim
        else:
            raise ValueError(f"Unsupported combination_mode: {self.combination_mode}")

        self.regressor = MLPBlocks(
            input_size=reg_input_size,
            output_size=1,
            hidden_size=mlp_hidden_size,
            num_layers=mlp_num_layers,
            dropout=mlp_dropout,
            use_residuals=mlp_use_residuals,
            normalization="batch"
        )

    def forward(
        self,
        mol_pyg_i: pyg.data.Data,
        mol_pyg_j: pyg.data.Data,
        ce_i=None, ce_j=None,
        ce_batch_idxs_i=None, ce_batch_idxs_j=None,
        prec_type_i=None, prec_type_j=None,
        inst_type_i=None, inst_type_j=None,
        **kwargs
    ):
        """
        Forward pass for Siamese GNN: encodes both molecules, pools, computes absolute difference, and regresses MCES.
        If CE, Prec, Inst are not provided, uses zeros as default.
        """
        device = mol_pyg_i.x.device

        # Default values for CE, Prec, Inst if not provided
        batch_size_i = mol_pyg_i.batch.max().item() + 1 if hasattr(mol_pyg_i, "batch") else 1
        batch_size_j = mol_pyg_j.batch.max().item() + 1 if hasattr(mol_pyg_j, "batch") else 1

        if ce_i is None:
            ce_i = th.zeros(batch_size_i, device=device)
        if ce_j is None:
            ce_j = th.zeros(batch_size_j, device=device)
        if ce_batch_idxs_i is None:
            ce_batch_idxs_i = th.arange(batch_size_i, device=device)
        if ce_batch_idxs_j is None:
            ce_batch_idxs_j = th.arange(batch_size_j, device=device)
        if prec_type_i is None:
            prec_type_i = th.zeros(batch_size_i, dtype=th.long, device=device)
        if prec_type_j is None:
            prec_type_j = th.zeros(batch_size_j, dtype=th.long, device=device)
        if inst_type_i is None:
            inst_type_i = th.zeros(batch_size_i, dtype=th.long, device=device)
        if inst_type_j is None:
            inst_type_j = th.zeros(batch_size_j, dtype=th.long, device=device)

        # Embeddings for molecule i
        ce_embed_i = self.embed_ce(ce_i, ce_batch_idxs_i, batch_size_i)
        prec_embed_i = self.embed_prec(prec_type_i)
        inst_embed_i = self.embed_inst(inst_type_i)
        mol_x_i = mol_pyg_i.x
        if self.ce_insert_location == "mol":
            mol_ce_embed = th.repeat_interleave(ce_embed_i, th.unique(mol_pyg_i.batch, return_counts=True)[1], dim=0)
            mol_x_i = th.cat([mol_x_i, mol_ce_embed], dim=1)
        if self.prec_insert_location == "mol":
            mol_prec_embed = th.repeat_interleave(prec_embed_i, th.unique(mol_pyg_i.batch, return_counts=True)[1], dim=0)
            mol_x_i = th.cat([mol_x_i, mol_prec_embed], dim=1)
        if self.inst_insert_location == "mol":
            mol_inst_embed = th.repeat_interleave(inst_embed_i, th.unique(mol_pyg_i.batch, return_counts=True)[1], dim=0)
            mol_x_i = th.cat([mol_x_i, mol_inst_embed], dim=1)

        # Embeddings for molecule j
        ce_embed_j = self.embed_ce(ce_j, ce_batch_idxs_j, batch_size_j)
        prec_embed_j = self.embed_prec(prec_type_j)
        inst_embed_j = self.embed_inst(inst_type_j)
        mol_x_j = mol_pyg_j.x
        if self.ce_insert_location == "mol":
            mol_ce_embed = th.repeat_interleave(ce_embed_j, th.unique(mol_pyg_j.batch, return_counts=True)[1], dim=0)
            mol_x_j = th.cat([mol_x_j, mol_ce_embed], dim=1)
        if self.prec_insert_location == "mol":
            mol_prec_embed = th.repeat_interleave(prec_embed_j, th.unique(mol_pyg_j.batch, return_counts=True)[1], dim=0)
            mol_x_j = th.cat([mol_x_j, mol_prec_embed], dim=1)
        if self.inst_insert_location == "mol":
            mol_inst_embed = th.repeat_interleave(inst_embed_j, th.unique(mol_pyg_j.batch, return_counts=True)[1], dim=0)
            mol_x_j = th.cat([mol_x_j, mol_inst_embed], dim=1)

        # Encode both molecules using the same GNN
        emb_i = self.mol_embedder(
            mol_x_i, mol_pyg_i.batch, mol_pyg_i.edge_index, mol_pyg_i.edge_attr
        )
        emb_j = self.mol_embedder(
            mol_x_j, mol_pyg_j.batch, mol_pyg_j.edge_index, mol_pyg_j.edge_attr
        )
        pooled_i = self.mol_pool(emb_i, mol_pyg_i.batch)
        pooled_j = self.mol_pool(emb_j, mol_pyg_j.batch)

        # Optionally add CE/Prec/Inst at MLP level
        mlp_parts_i = [pooled_i]
        mlp_parts_j = [pooled_j]
        if self.ce_insert_location == "mlp":
            mlp_parts_i.append(ce_embed_i)
            mlp_parts_j.append(ce_embed_j)
        if self.prec_insert_location == "mlp":
            mlp_parts_i.append(prec_embed_i)
            mlp_parts_j.append(prec_embed_j)
        if self.inst_insert_location == "mlp":
            mlp_parts_i.append(inst_embed_i)
            mlp_parts_j.append(inst_embed_j)
        mlp_input_i = th.cat(mlp_parts_i, dim=1)
        mlp_input_j = th.cat(mlp_parts_j, dim=1)

        # Combine the two MLP inputs according to the requested mode.
        # - 'abs_diff': order-invariant absolute element-wise difference (default)
        # - 'signed_diff': order-sensitive signed difference (xi - xj)
        # - 'concat_avg': symmetric concat: avg( reg([xi,xj]), reg([xj,xi]) )
        # - 'concat': simple concat [xi, xj] (order-sensitive)
        if self.combination_mode == "abs_diff":
            combined = th.abs(mlp_input_i - mlp_input_j)
            pred_mces = self.regressor(combined)
            return pred_mces.squeeze(-1)
        elif self.combination_mode == "signed_diff":
            combined = mlp_input_i - mlp_input_j
            pred_mces = self.regressor(combined)
            return pred_mces.squeeze(-1)
        elif self.combination_mode == "concat_avg":
            # Build concatenated inputs for both orders, run regressor and average.
            cat_ij = th.cat([mlp_input_i, mlp_input_j], dim=1)
            cat_ji = th.cat([mlp_input_j, mlp_input_i], dim=1)
            pred1 = self.regressor(cat_ij)
            pred2 = self.regressor(cat_ji)
            pred = 0.5 * (pred1 + pred2)
            return pred.squeeze(-1)
        elif self.combination_mode == "concat":
            cat = th.cat([mlp_input_i, mlp_input_j], dim=1)
            pred = self.regressor(cat)
            return pred.squeeze(-1)
        elif self.combination_mode == "cross_attn":
            # cross-attention between the two pooled embeddings.
            # We compute attention of i->j and j->i separately and include both
            # attended vectors alongside the original embeddings, producing
            # a combined vector [xi, attn_ij, xj, attn_ji]. This is expressive
            # and we keep symmetry by computing attention in both directions.
            # If batch sizes differ we fall back to simple concat (safe fallback).
            if batch_size_i != batch_size_j:
                # fallback: concatenate (note: typically batches for pair datasets are aligned)
                print(f"Warning: cross_attn requested but batch sizes differ ({batch_size_i}!={batch_size_j}), falling back to concat")
                cat = th.cat([mlp_input_i, mlp_input_j], dim=1)
                pred = self.regressor(cat)
                return pred.squeeze(-1)

            # project into attention embedding space and run MHA (seq_len=1)
            proj_i = self.cross_proj(mlp_input_i)
            proj_j = self.cross_proj(mlp_input_j)
            proj_i_seq = proj_i.unsqueeze(1)  # (B, 1, D)
            proj_j_seq = proj_j.unsqueeze(1)
            attn_ij, _ = self.cross_mha(query=proj_i_seq, key=proj_j_seq, value=proj_j_seq)
            attn_ji, _ = self.cross_mha(query=proj_j_seq, key=proj_i_seq, value=proj_i_seq)
            attn_ij = attn_ij.squeeze(1)
            attn_ji = attn_ji.squeeze(1)
            combined = th.cat([mlp_input_i, attn_ij, mlp_input_j, attn_ji], dim=1)
            pred = self.regressor(combined)
            return pred.squeeze(-1)
        else:
            raise ValueError(f"Unsupported combination_mode: {self.combination_mode}")

    def _compute_mol_feats_sizes(self):
        """Compute molecular feature sizes (does not rely on model parameters)."""
        self.mol_node_feats_size, self.mol_edge_feats_size = get_mol_feats_sizes(
            self.mol_node_feats,
            self.mol_edge_feats,
            self.mol_pe_embed_k
        )

    def _ce_location_check(self):
        # input not used, just for shape consistency
        pass

    def _prec_location_check(self):
        # input not used, just for shape consistency
        pass

    def _inst_location_check(self):
        # input not used, just for shape consistency
        pass 