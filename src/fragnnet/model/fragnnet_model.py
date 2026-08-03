"""
FragGNNMet Model for MS/MS Spectrum Prediction
"""

import logging

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as th_checkpoint
import torch_geometric as pyg

from fragnnet.model.base_model import (
    AdductScaler,
    CEModel,
    CEScaler,
    ContextEmbedding,
    CutBondEncoder,
    FragModeModel,
    FragModeScaler,
    InstModel,
    PrecModel,
    StratumScaler,
)
from fragnnet.model.form_embedder import get_embedder
from fragnnet.model.nn_blocks import GNN, MLPBlocks, build_pool_module
from fragnnet.utils.data_utils import combine_formulae
from fragnnet.utils.feat_utils import get_mol_feats_sizes
from fragnnet.utils.formula_utils import PREC_TYPE_TO_FORMULA_DIFF
from fragnnet.utils.frag_utils import (
    ELEMENT_TO_VE,
    HEAVY_ELEMENTS,
    compute_boundary_mask,
    get_edge_feats,
    get_node_feats,
    th_long_to_mask,
)
from fragnnet.utils.misc_utils import (
    LOG_ZERO,
    check_pyg_compile,
    check_pyg_full_compile,
    scatter_logsoftmax,
    scatter_logsumexp,
    scatter_masked_softmax,
    scatter_reduce,
)
from fragnnet.utils.spec_utils import batched_bin_func

logger = logging.getLogger(__name__)


class FraGNNetModel(nn.Module, CEModel, PrecModel, InstModel, FragModeModel):
    def __init__(
        self,
        num_depth: int,
        num_hs: int,
        num_elements: int,
        int_embedder: str,
        int_embedder_tight: bool,
        mol_node_feats: list[str],
        mol_edge_feats: list[str],
        mol_pe_embed_k: int,
        mol_hidden_size: int,
        mol_num_layers: int,
        mol_gnn_type: str,
        mol_normalization: str,
        mol_dropout: float,
        mol_num_heads: int,
        mol_pool_type: str,
        frag_node_feats: list[str],
        frag_edge_feats: list[str],
        frag_hidden_size: int,
        frag_num_layers: int,
        frag_gnn_type: str,
        frag_normalization: str,
        frag_dropout: float,
        frag_pool_type: str,
        frag_embed_combine: str,
        frag_pool_combine: str,
        mlp_output_format: str,
        mlp_hidden_size: int,
        mlp_normalization: str,
        mlp_dropout: float,
        mlp_num_layers: int,
        mlp_use_residuals: bool,
        cc_interstage_type: str,
        cc_interstage_use_rest: bool,
        nb_iso: bool,
        skip_edge_loss: bool,
        mask_null_formula: bool,
        predict_oos: bool,
        bin_output: bool,
        mz_bin_res: float,
        mz_max: float,
        ce_insert_location: str,
        ce_insert_type: str,
        ce_insert_merge: bool,
        ce_insert_size: int,
        nce_max: float,
        nce_mean: float,
        nce_std: float,
        prec_insert_location: str,
        prec_insert_size: int,
        prec_types: list[str],
        inst_insert_location: str,
        inst_insert_size: int,
        inst_types: list[str],
        frag_mode_insert_location: str,
        frag_mode_insert_size: int,
        frag_modes: list[str],
        frag_mode_scale: bool,
        ce_scale: bool,
        ce_scaler_hidden_dim: int,
        adduct_scale: bool,
        stratum_scale: bool,
        stratum_scaler_embed_dim: int,
        output_formula_str: bool,
        use_nce: bool,
        use_ace: bool,
        ace_max: float,
        ace_mean: float,
        ace_std: float,
        cc_feature_dropout: float,
        cc_atom_dropout: float = 0.0,
        cc_channel_dropout: float = 0.0,
        depth_ce_bias: bool = False,
        context_embedding: bool = False,
        context_embedding_dim: int = 64,
        context_embedding_hidden_dim: int = 64,
        mol_msg_gate_mode: str = "static",
        mol_msg_gate_hidden_dim: int = 64,
        cut_bond_hidden_dim: int = 256,
        context_insert_location: str = "mol,mlp",
        debug_validate_outputs: bool = False,
        skip_explainability_when_training: bool = False,
        candidate_ion_head: bool = False,
        candidate_ion_formula_features: bool = True,
        candidate_ion_neutral_loss_features: bool = True,
        candidate_ion_chunk_size: int = 32768,
        candidate_ion_checkpoint: bool = True,
        ion_channel_factorized: bool = False,
        ion_channel_adduct_gated: bool = False,
        ion_channel_rank: int = 32,
    ):
        # nn.Module init
        super().__init__()

        self._ce_init(
            int_embedder=int_embedder,
            ce_insert_location=ce_insert_location,
            ce_insert_type=ce_insert_type,
            ce_insert_merge=ce_insert_merge,
            ce_insert_size=ce_insert_size,
            nce_max=nce_max,
            nce_mean=nce_mean,
            nce_std=nce_std,
            use_nce=use_nce,
            use_ace=use_ace,
            ace_max=ace_max,
            ace_mean=ace_mean,
            ace_std=ace_std,
        )

        self._prec_init(
            prec_insert_location=prec_insert_location,
            prec_insert_size=prec_insert_size,
            prec_num_types=len(prec_types),
        )

        self._inst_init(
            inst_insert_location=inst_insert_location,
            inst_insert_size=inst_insert_size,
            inst_num_types=len(inst_types),
        )

        self._frag_mode_init(
            frag_mode_insert_location=frag_mode_insert_location,
            frag_mode_insert_size=frag_mode_insert_size,
            frag_mode_num_types=len(frag_modes),
        )
        self.frag_modes = frag_modes

        # Combined context (stratum) embedding. When enabled, CE/adduct/instrument/frag_mode are
        # combined into ONE learned vector injected at context_insert_location; the individual
        # *_insert paths are subsumed (their embeddings feed the context and are NOT injected
        # separately, so their mol/mlp input contributions are zeroed here). See
        # docs/plans/conditioning_refactor_plan.md. Gated: off (default) leaves existing behavior intact.
        self.context_embedding = context_embedding
        if context_embedding:
            active_dims = [
                self.ce_embed_dim if self.ce_insert_locations else 0,
                prec_insert_size if prec_insert_location != "none" else 0,
                inst_insert_size if inst_insert_location != "none" else 0,
                frag_mode_insert_size if frag_mode_insert_location != "none" else 0,
            ]
            ctx_in = sum(active_dims)
            if ctx_in == 0:
                raise ValueError("context_embedding=True requires at least one metadata enabled")
            self.context_insert_locations = {loc.strip() for loc in context_insert_location.split(",")}
            invalid = self.context_insert_locations - {"mol", "mlp"}
            if invalid:
                raise ValueError(f"context_insert_location must be mol/mlp, got {invalid}")
            self.context_embedder = ContextEmbedding(
                ctx_in, context_embedding_hidden_dim, context_embedding_dim
            )
            # context replaces the per-metadata injections -> zero their input-dim contributions
            self.ce_mol_input_dim = self.ce_mlp_input_dim = 0
            self.prec_mol_input_dim = self.prec_mlp_input_dim = 0
            self.inst_mol_input_dim = self.inst_mlp_input_dim = 0
            self.frag_mode_mol_input_dim = self.frag_mode_mlp_input_dim = 0
            self.context_mol_input_dim = (
                context_embedding_dim if "mol" in self.context_insert_locations else 0
            )
            self.context_mlp_input_dim = (
                context_embedding_dim if "mlp" in self.context_insert_locations else 0
            )
        else:
            self.context_embedder = None
            self.context_insert_locations = set()
            self.context_mol_input_dim = 0
            self.context_mlp_input_dim = 0

        self.num_depth = num_depth
        self.num_hs = num_hs
        self.num_elements = num_elements
        self.debug_validate_outputs = debug_validate_outputs
        # When True, the frag_real_* explainability marginals (P(n), P(f), P(f|n), P(n|f), P(h) and
        # the nb_iso variants) are skipped during TRAINING forward passes -- they are consumed only
        # by the entropy aux losses and by eval/inference, so with those off they are dead weight on
        # the forward peak. Only affects self.training=True passes; eval/inference always compute
        # them. Default False keeps the original behavior. The PL module gates this on the active
        # loss set so it is never enabled while a consumer is live.
        self.skip_explainability_when_training = skip_explainability_when_training
        if not 0.0 <= cc_feature_dropout <= 1.0:
            raise ValueError(f"cc_feature_dropout must be in [0, 1], got {cc_feature_dropout}")
        self.cc_feature_dropout = float(cc_feature_dropout)
        # Per-ATOM dropout on the cc mask during training: independently drop each atom from a
        # fragment's atom-membership set, so the pooled cc embedding sees a random subset of the
        # fragment's atoms. Regularizes against memorizing exact atom sets (targets the train->val
        # gap). Distinct from cc_feature_dropout, which zeroes the WHOLE cc group for the whole
        # batch on a single scalar draw. Applied only to the local cc_mask, so boundary_cc /
        # boundary_pair keep the true mask (same convention as cc_corruption_frac). A fragment is
        # never emptied: any fragment that loses all atoms is restored.
        if not 0.0 <= cc_atom_dropout <= 1.0:
            raise ValueError(f"cc_atom_dropout must be in [0, 1], got {cc_atom_dropout}")
        self.cc_atom_dropout = float(cc_atom_dropout)
        if not 0.0 <= cc_channel_dropout <= 1.0:
            raise ValueError(f"cc_channel_dropout must be in [0, 1], got {cc_channel_dropout}")
        self.cc_channel_dropout = float(cc_channel_dropout)
        # Diagnostic-only: fraction of cc mask bits to randomly flip at eval time, to test
        # whether the model relies on precise cc atom-identity content. Not a config option;
        # set directly on the model instance (see scripts/debug/debug_cc_corruption_eval.py).
        self.cc_corruption_frac = 0.0
        # Diagnostic-only: bypass molecular message passing while retaining the learned input
        # projection and exact cc atom mask. This removes surrounding-molecule context from each
        # atom embedding without changing checkpoint shapes.
        self.mol_context_input_only = False

        # calculate node/edge feats sizes
        self.mol_node_feats = mol_node_feats
        self.mol_edge_feats = mol_edge_feats
        self.mol_pe_embed_k = mol_pe_embed_k
        self._compute_mol_feats_sizes()

        # setup mol gnn
        self.mol_node_feats_size += (
            self.ce_mol_input_dim
            + self.prec_mol_input_dim
            + self.inst_mol_input_dim
            + self.frag_mode_mol_input_dim
            + self.context_mol_input_dim
        )
        mol_kwargs = {
            "node_feats_size": self.mol_node_feats_size,
            "edge_feats_size": self.mol_edge_feats_size,
            "hidden_size": mol_hidden_size,
            "num_layers": mol_num_layers,
            "gnn_type": mol_gnn_type,
            "dropout": mol_dropout,
            "normalization": mol_normalization,
            "num_heads": mol_num_heads,
        }
        # Context-gated mol messages: the gate is conditioned on the combined context vector, so
        # it needs context_embedding to have produced one. Metadata concatenated at the mol input
        # cannot serve -- it is per-atom-broadcast and already inside x.
        self.mol_msg_gate = mol_gnn_type == "GINE_GATED"
        if self.mol_msg_gate:
            if not self.context_embedding:
                raise ValueError(
                    "mol_gnn_type='GINE_GATED' requires context_embedding=True to supply the "
                    "context vector that conditions the gate"
                )
            mol_kwargs["ctx_dim"] = context_embedding_dim
            mol_kwargs["gate_mode"] = mol_msg_gate_mode
            mol_kwargs["gate_hidden_dim"] = mol_msg_gate_hidden_dim

        # Mol GNN
        self.mol_embedder = GNN(**mol_kwargs)
        self.mol_pool_type = mol_pool_type
        self.mol_pool = build_pool_module(mol_pool_type, mol_hidden_size)
        if mol_pool_type in ["mean_max_sum", "mean_std_softmax"]:
            mol_pool_out_size = mol_hidden_size * 3
        else:
            mol_pool_out_size = mol_hidden_size

        if int_embedder_tight:
            formula_d = {"max_count_int": 255}
            depth_d = {"max_count_int": num_depth + 1}
            complement_d = {"max_count_int": 2}
            dag_degree_d = {"max_count_int": 64}
            num_heavy_atoms_d = {"max_count_int": 128}
            # cut bonds per fragment: the SMARTS ring-cleavage rules are the heaviest at 4 cuts
            num_cut_bonds_d = {"max_count_int": 32}
            # 2*RDBE (see below); ~2*(num_heavy_atoms+1) at most
            rdbe_d = {"max_count_int": 256}
        else:
            formula_d = depth_d = complement_d = dag_degree_d = num_heavy_atoms_d = {}
            num_cut_bonds_d = rdbe_d = {}
        self.formula_embedder = get_embedder(int_embedder, **formula_d)
        self.depth_embedder = get_embedder(int_embedder, **depth_d)
        self.complement_embedder = get_embedder(int_embedder, **complement_d)
        self.out_degree_embedder = get_embedder(int_embedder, **dag_degree_d)
        self.in_degree_embedder = get_embedder(int_embedder, **dag_degree_d)
        self.num_heavy_atoms_embedder = get_embedder(int_embedder, **num_heavy_atoms_d)
        # Built only when enabled (the frag_q precedent below), unlike the embedders above: an
        # unconditional module would add dead parameters to every existing config and break
        # strict=True checkpoint loads for runs that predate these features.
        if "num_cut_bonds" in frag_node_feats:
            self.num_cut_bonds_embedder = get_embedder(int_embedder, **num_cut_bonds_d)
        if "rdbe" in frag_node_feats:
            self.rdbe_embedder = get_embedder(int_embedder, **rdbe_d)
        if "frag_q" in frag_node_feats:
            self.frag_q_proj = nn.Linear(1, self.out_degree_embedder.num_dim)
        if "rdbe" in frag_node_feats:
            # Validate at construction rather than on the first batch: these are config errors and
            # should not surface hours into a run.
            if "base_formula" not in frag_node_feats:
                raise ValueError("rdbe requires base_formula in frag_node_feats")
            if "h_counts" not in frag_node_feats:
                raise ValueError(
                    "rdbe requires h_counts in frag_node_feats (base_formula has no H column); "
                    "h_counts adds a DAG column but contributes 0 feature dims"
                )
            # Ring-and-double-bond equivalents: RDBE = 1 + sum_i n_i * (v_i - 2) / 2, the standard
            # generalization of "C - H/2 + N/2 + 1" to arbitrary valences. Derived from
            # ELEMENT_TO_VE rather than hardcoded so it stays correct if the element list changes.
            # H is appended last in ELEMENTS and is monovalent. Stored doubled (v_i - 2, not
            # (v_i - 2)/2) so the contraction stays in integer arithmetic; see the forward.
            _val = th.tensor(
                [ELEMENT_TO_VE[e] for e in HEAVY_ELEMENTS] + [1], dtype=th.long
            )
            if _val.shape[0] != self.num_elements:
                raise ValueError(
                    f"rdbe valence table has {_val.shape[0]} entries but num_elements="
                    f"{self.num_elements}; they must describe the same element list"
                )
            self.register_buffer("_rdbe_coef_buf", _val - 2, persistent=False)
        self.frag_node_feats = frag_node_feats
        self.frag_edge_feats = frag_edge_feats
        self._compute_frag_feats_sizes()
        self.mlp_output_dim = 2 * self.num_hs + 1
        if "cmf_h_formulae_idx" in self.frag_node_feats:
            # if cmf, we need to double the output dim
            self.mlp_output_dim *= 2

        # define interstage
        assert cc_interstage_type in [
            "add",
            "sub",
            "linear",
            "direct",
            "gated",
            "mlp",
            "bilinear",
        ]
        self.cc_interstage_type = cc_interstage_type
        if self.cc_interstage_type == "linear":
            self.cc_interstage = nn.Linear(mol_pool_out_size * 2, mol_pool_out_size)
        elif self.cc_interstage_type == "gated":
            # Gated fusion: learns to mix frag and mol embeddings
            self.cc_interstage = nn.Sequential(
                nn.Linear(mol_pool_out_size * 2, mol_pool_out_size), nn.Sigmoid()
            )
        elif self.cc_interstage_type == "mlp":
            # Deep fusion using MLPBlocks (more flexible, parameter-efficient with residuals)
            self.cc_interstage = MLPBlocks(
                input_size=mol_pool_out_size * 2,
                output_size=mol_pool_out_size,
                hidden_size=mlp_hidden_size,
                num_layers=2,
                dropout=mlp_dropout,
                use_residuals=mlp_use_residuals,
                normalization=mlp_normalization,
            )
        elif self.cc_interstage_type == "bilinear":
            # Bilinear interaction
            self.cc_interstage_bilinear = nn.Bilinear(
                mol_pool_out_size, mol_pool_out_size, mol_pool_out_size
            )
            self.cc_interstage_proj = nn.Linear(mol_pool_out_size, mol_pool_out_size)

        if "boundary_pair" in self.frag_node_feats:
            # Projects cat(inside_atom_embed, outside_atom_embed) → mol_hidden_size.
            # Pooling over pairs then yields mol_pool_out_size (same as cc / boundary_cc).
            # TODO(next version): remove the legacy boundary_pair feature path entirely.
            # NOTE: being linear, this collapses under pooling to
            # Linear(mean(inside_atoms), mean(outside_atoms)) -- the bond pairing is discarded.
            # `cut_bond` below is the non-degenerate replacement at identical output width.
            self.boundary_pair_proj = nn.Linear(mol_hidden_size * 2, mol_hidden_size)
        if "cut_bond" in self.frag_node_feats:
            # Non-degenerate replacement for boundary_pair: adds an explicit in*out interaction
            # and a hidden nonlinearity so the bond pairing survives pooling. Same output width
            # (mol_pool_out_size) and same pooling as boundary_pair, so swapping the two feature
            # names is a clean one-variable comparison. Reuses the SAME precomputed
            # boundary_pair_* indices, so no DAG regeneration is required.
            self.cut_bond_encoder = CutBondEncoder(
                atom_dim=mol_hidden_size,
                hidden_dim=cut_bond_hidden_dim,
                out_dim=mol_hidden_size,
            )

        frag_kwargs = {
            "node_feats_size": self.frag_node_feats_size,
            "edge_feats_size": self.frag_edge_feats_size,
            "hidden_size": frag_hidden_size,
            "num_layers": frag_num_layers,
            "gnn_type": frag_gnn_type,
            "dropout": frag_dropout,
            "normalization": frag_normalization,
        }

        self.frag_embedder = GNN(**frag_kwargs)
        self.frag_pool_type = frag_pool_type
        self.frag_pool = build_pool_module(frag_pool_type, frag_hidden_size)
        self.frag_embed_combine = frag_embed_combine
        self.frag_pool_combine = frag_pool_combine
        self.frag_pool_preserves_hidden_size = self.frag_pool_type not in [
            "mean_max_sum",
            "mean_std_softmax",
        ]
        if (
            self.frag_pool_combine in ["subtract", "add"]
            and not self.frag_pool_preserves_hidden_size
        ):
            raise ValueError(
                "frag_pool_combine must be 'none' when frag_pool_type expands feature width. "
                f"Received frag_pool_type={self.frag_pool_type!r}, "
                f"frag_pool_combine={self.frag_pool_combine!r}."
            )
        self.mlp_output_format = mlp_output_format

        # Pool output width (multi-stat pools triple the feature width). Hoisted here so both the
        # concat set-context injection and the OOS head reuse a single definition.
        self.frag_pool_out_size = self.frag_embedder.hidden_size * (
            3 if self.frag_pool_type in ["mean_max_sum", "mean_std_softmax"] else 1
        )

        # determine mlp input dims
        if self.frag_embed_combine == "cat":
            mlp_input_dim = 2 * self.frag_embedder.hidden_size
        else:
            assert self.frag_embed_combine == "avg", self.frag_embed_combine
            mlp_input_dim = self.frag_embedder.hidden_size
        mlp_input_dim += (
            self.ce_mlp_input_dim
            + self.prec_mlp_input_dim
            + self.inst_mlp_input_dim
            + self.frag_mode_mlp_input_dim
            + self.context_mlp_input_dim
        )
        if self.frag_pool_combine == "concat":
            if self.mlp_output_format != "formula":
                raise ValueError(
                    "frag_pool_combine='concat' is only supported with "
                    f"mlp_output_format='formula', got {self.mlp_output_format!r}."
                )
            # Append the pooled set-context to each node's readout input (DeepSets-style global
            # feature); the readout MLP then learns an arbitrary node x set interaction instead of
            # the fixed elementwise shift that 'subtract'/'add' apply.
            mlp_input_dim += self.frag_pool_out_size

        self.candidate_ion_head = candidate_ion_head
        self.candidate_ion_formula_features = candidate_ion_formula_features
        self.candidate_ion_neutral_loss_features = candidate_ion_neutral_loss_features
        self.candidate_ion_chunk_size = int(candidate_ion_chunk_size)
        self.candidate_ion_checkpoint = bool(candidate_ion_checkpoint)
        self.candidate_ion_feature_dim = 0
        self.ion_channel_factorized = bool(ion_channel_factorized)
        self.ion_channel_adduct_gated = bool(ion_channel_adduct_gated)
        self.ion_channel_rank = int(ion_channel_rank)
        if self.candidate_ion_head and (self.ion_channel_factorized or self.ion_channel_adduct_gated):
            raise ValueError(
                "candidate_ion_head and ion_channel_factorized/ion_channel_adduct_gated are "
                "mutually exclusive"
            )
        if self.ion_channel_factorized and self.ion_channel_adduct_gated:
            raise ValueError(
                "ion_channel_factorized and ion_channel_adduct_gated are mutually exclusive"
            )
        if self.candidate_ion_head:
            # These features describe the candidate peak row, not just the fragment node. The
            # candidate scores are reshaped back to (node, H-channel), so downstream spectrum
            # normalization and metric code stay on the existing path.
            self.candidate_h_channel_embedder = get_embedder(
                int_embedder, max_count_int=self.mlp_output_dim
            )
            self.candidate_cmf_flag_embedder = get_embedder(int_embedder, max_count_int=2)
            self.candidate_ion_feature_dim += (
                self.candidate_h_channel_embedder.num_dim
                + self.candidate_cmf_flag_embedder.num_dim
            )
            if self.candidate_ion_formula_features:
                if (
                    "base_formula" not in self.frag_node_feats
                    or "h_counts" not in self.frag_node_feats
                ):
                    raise ValueError(
                        "candidate_ion_formula_features requires base_formula and h_counts in "
                        "frag_node_feats"
                    )
                self.candidate_ion_feature_dim += (
                    self.num_elements * self.formula_embedder.num_dim
                )
            if self.candidate_ion_neutral_loss_features:
                if not self.candidate_ion_formula_features:
                    raise ValueError(
                        "candidate_ion_neutral_loss_features requires "
                        "candidate_ion_formula_features=True"
                    )
                if "depth" not in self.frag_node_feats:
                    raise ValueError(
                        "candidate_ion_neutral_loss_features requires depth in frag_node_feats"
                    )
                self.candidate_ion_feature_dim += (
                    self.num_elements * self.formula_embedder.num_dim
                )
        else:
            self.candidate_h_channel_embedder = None
            self.candidate_cmf_flag_embedder = None

        if self.ion_channel_factorized or self.ion_channel_adduct_gated:
            if self.ion_channel_rank <= 0:
                raise ValueError(f"ion_channel_rank must be positive, got {ion_channel_rank}")
            self.ion_channel_node_proj = nn.Linear(
                mlp_input_dim, self.ion_channel_rank, bias=False
            )
            self.ion_channel_embed = nn.Embedding(self.mlp_output_dim, self.ion_channel_rank)
            # Identity start: residual is zero, but node_proj receives gradients immediately
            # because ion_channel_embed is nonzero.
            nn.init.zeros_(self.ion_channel_node_proj.weight)
            nn.init.normal_(self.ion_channel_embed.weight, mean=0.0, std=0.02)
            if self.ion_channel_adduct_gated:
                # Adduct-gated: per-adduct multiplicative gate on the rank-r node factor gives a
                # node x channel x adduct trilinear residual (the interaction AdductScaler's
                # node-invariant affine cannot express). No bias here: AdductScaler owns the
                # adduct x channel additive term. Gate init at ones keeps the identity start
                # (residual still zero because node_proj is zero) while preserving node_proj grads.
                self.ion_channel_adduct_gate = nn.Embedding(
                    len(prec_types) + 1, self.ion_channel_rank
                )
                nn.init.ones_(self.ion_channel_adduct_gate.weight)
                self.ion_channel_bias = None
            else:
                self.ion_channel_adduct_gate = None
                self.ion_channel_bias = nn.Parameter(th.zeros(self.mlp_output_dim))
        else:
            self.ion_channel_node_proj = None
            self.ion_channel_embed = None
            self.ion_channel_bias = None
            self.ion_channel_adduct_gate = None

        #  mlp for formula
        if self.mlp_output_format in ["formula", "node_formula"]:
            formula_mlp_kwargs = {
                "input_size": mlp_input_dim + self.candidate_ion_feature_dim,
                "output_size": 1 if self.candidate_ion_head else self.mlp_output_dim,
                "hidden_size": mlp_hidden_size,
                "num_layers": mlp_num_layers,
                "dropout": mlp_dropout,
                "use_residuals": mlp_use_residuals,
                "normalization": mlp_normalization,
            }
            self.formula_module = MLPBlocks(**formula_mlp_kwargs)

        # mode-conditioned output scaling (applied after formula_module)
        self.frag_mode_scale = frag_mode_scale
        if self.frag_mode_scale:
            self.frag_mode_scaler = FragModeScaler(
                num_frag_modes=len(frag_modes),
                output_dim=self.mlp_output_dim,
            )
        else:
            self.frag_mode_scaler = None

        # CE-conditioned FiLM output scaling (applied after frag_mode_scaler)
        self.ce_scale = ce_scale
        if self.ce_scale:
            self.ce_scaler = CEScaler(
                nce_mean=nce_mean,
                nce_std=nce_std,
                use_nce=use_nce,
                ace_mean=ace_mean,
                ace_std=ace_std,
                use_ace=use_ace,
                output_dim=self.mlp_output_dim,
                hidden_dim=ce_scaler_hidden_dim,
            )
        else:
            self.ce_scaler = None

        # adduct-type FiLM output scaling (applied after ce_scaler)
        self.adduct_scale = adduct_scale
        if self.adduct_scale:
            self.adduct_scaler = AdductScaler(
                num_adduct_types=len(prec_types),
                output_dim=self.mlp_output_dim,
            )
        else:
            self.adduct_scaler = None

        # combined stratum FiLM output scaling (applied after adduct_scaler)
        self.stratum_scale = stratum_scale
        if self.stratum_scale:
            self.stratum_scaler = StratumScaler(
                num_inst_types=len(inst_types),
                num_frag_modes=len(frag_modes),
                num_adduct_types=len(prec_types),
                embed_dim=stratum_scaler_embed_dim,
                output_dim=self.mlp_output_dim,
            )
        else:
            self.stratum_scaler = None

        # CE × Depth Prior: per-depth logit bias conditioned on CE (§3c)
        self.depth_ce_bias = depth_ce_bias
        if depth_ce_bias:
            if self.ce_embed_dim == 0:
                raise ValueError("depth_ce_bias requires CE to be enabled (use_nce or use_ace)")
            if "depth" not in frag_node_feats:
                raise ValueError("depth_ce_bias requires 'depth' in frag_params.pyg_node_feats")
            self.depth_ce_bias_layer = nn.Linear(self.ce_embed_dim, num_depth + 1)
            nn.init.zeros_(self.depth_ce_bias_layer.weight)
            nn.init.zeros_(self.depth_ce_bias_layer.bias)
        else:
            self.depth_ce_bias_layer = None

        # use extra mlp for node formula
        if self.mlp_output_format in ["node_formula"]:
            node_mlp_kwargs = {
                "input_size": 2 * self.frag_embedder.hidden_size,
                "output_size": 1,
                "hidden_size": mlp_hidden_size,
                "num_layers": mlp_num_layers,
                "dropout": mlp_dropout,
                "use_residuals": mlp_use_residuals,
                "normalization": mlp_normalization,
            }
            self.node_module = MLPBlocks(**node_mlp_kwargs)
        else:
            self.node_module = None

        self.predict_oos = predict_oos
        if self.predict_oos:
            # Account for pooling output sizes
            frag_pool_out_size = self.frag_pool_out_size

            oos_mlp_kwargs = {
                "input_size": mol_pool_out_size + frag_pool_out_size,
                "output_size": 1,
                "hidden_size": mlp_hidden_size,
                "num_layers": mlp_num_layers,
                "dropout": mlp_dropout,
                "use_residuals": mlp_use_residuals,
                "normalization": "none",  # never normalized
            }
            self.oos_module = MLPBlocks(**oos_mlp_kwargs)
        else:
            self.oos_module = None

        self.skip_edge_loss = skip_edge_loss
        self.mask_null_formula = mask_null_formula
        self.nb_iso = nb_iso
        self.bin_output = bin_output
        self.mz_bin_res = mz_bin_res
        self.mz_max = mz_max
        self.output_formula_str = output_formula_str
        self.cc_interstage_use_rest = cc_interstage_use_rest

        if self.bin_output:
            self.register_buffer("mz_bins", th.arange(mz_bin_res, mz_max + mz_bin_res, mz_bin_res), persistent=False)

        # this is required
        assert "h_formulae_idx" in self.frag_node_feats

        # Pre-compute h_counts as a buffer — fixed for the lifetime of the model.
        # Layout: [0, -1, +1, -2, +2, ..., -num_hs, +num_hs]
        _h_counts = th.zeros([2 * num_hs + 1], dtype=th.long)
        _h_counts[1 + 2 * th.arange(num_hs)] = -th.arange(1, num_hs + 1)
        _h_counts[2 + 2 * th.arange(num_hs)] = th.arange(1, num_hs + 1)
        include_cmf = "cmf_h_formulae_idx" in frag_node_feats
        if include_cmf:
            _h_counts = th.cat([_h_counts, _h_counts], dim=0)
        self.register_buffer("_h_counts_buf", _h_counts, persistent=False)
        _h_counts_range = th.arange(-num_hs, num_hs + 1, dtype=th.long)
        if include_cmf:
            _h_counts_range = th.cat([_h_counts_range, _h_counts_range])
        self.register_buffer("_h_counts_range_buf", _h_counts_range, persistent=False)

    def _maybe_apply_cc_feature_dropout(self, frag_node_mask_embed: th.Tensor) -> th.Tensor:
        """Drop the full-fragment cc feature group for a whole training batch."""
        if not self.training or self.cc_feature_dropout <= 0.0 or "cc" not in self.frag_node_feats:
            return frag_node_mask_embed
        keep = th.rand((), device=frag_node_mask_embed.device) >= self.cc_feature_dropout
        return frag_node_mask_embed * keep.to(dtype=frag_node_mask_embed.dtype)

    def _maybe_apply_cc_channel_dropout(self, frag_node_mask_embed: th.Tensor) -> th.Tensor:
        """Drop pooled cc embedding channels with one mask shared by all fragments."""
        if not self.training or self.cc_channel_dropout <= 0.0:
            return frag_node_mask_embed
        if self.cc_channel_dropout == 1.0:
            return th.zeros_like(frag_node_mask_embed)
        keep = (
            th.rand(
                1,
                frag_node_mask_embed.shape[1],
                device=frag_node_mask_embed.device,
            )
            >= self.cc_channel_dropout
        )
        return (
            frag_node_mask_embed
            * keep.to(dtype=frag_node_mask_embed.dtype)
            / (1.0 - self.cc_channel_dropout)
        )

    def _ce_location_check(self):
        assert "frag" not in self.ce_insert_locations, "ce_insert_location=frag not supported"

    def _prec_location_check(self):
        assert self.prec_insert_location != "frag", "prec_insert_location=frag not supported"

    def _inst_location_check(self):
        assert self.inst_insert_location != "frag", "inst_insert_location=frag not supported"

    def _frag_mode_location_check(self):
        assert self.frag_mode_insert_location != "frag", (
            "frag_mode_insert_location=frag not supported"
        )

    def _compute_mol_feats_sizes(self):
        """method compute mol feature size
        these features don't rely on any model parameters
        """
        self.mol_node_feats_size, self.mol_edge_feats_size = get_mol_feats_sizes(
            self.mol_node_feats, self.mol_edge_feats, self.mol_pe_embed_k
        )

    def _compute_frag_feats_sizes(self):
        """method compute frag-graph feature size
        these features do depend on model parameters
        """
        # Determine the actual output size after pooling
        if self.mol_pool_type in ["mean_max_sum", "mean_std_softmax"]:
            mol_pool_out_size = self.mol_embedder.hidden_size * 3
        else:
            mol_pool_out_size = self.mol_embedder.hidden_size

        # nodes
        self.frag_node_feats_size = 0
        if "cc" in self.frag_node_feats:
            self.frag_node_feats_size += mol_pool_out_size
        if "boundary_cc" in self.frag_node_feats:
            self.frag_node_feats_size += mol_pool_out_size
        if "boundary_pair" in self.frag_node_feats:
            self.frag_node_feats_size += mol_pool_out_size
        if "cut_bond" in self.frag_node_feats:
            self.frag_node_feats_size += mol_pool_out_size
        if "base_formula" in self.frag_node_feats:
            self.frag_node_feats_size += self.num_elements * self.formula_embedder.num_dim
        if "neutral_loss_formula" in self.frag_node_feats:
            self.frag_node_feats_size += self.num_elements * self.formula_embedder.num_dim
        if "depth" in self.frag_node_feats:
            self.frag_node_feats_size += self.num_depth * self.depth_embedder.num_dim
        if "out_degree" in self.frag_node_feats:
            self.frag_node_feats_size += self.out_degree_embedder.num_dim
        if "in_degree" in self.frag_node_feats:
            self.frag_node_feats_size += self.in_degree_embedder.num_dim
        if "num_heavy_atoms" in self.frag_node_feats:
            self.frag_node_feats_size += self.num_heavy_atoms_embedder.num_dim
        if "num_cut_bonds" in self.frag_node_feats:
            self.frag_node_feats_size += self.num_cut_bonds_embedder.num_dim
        if "rdbe" in self.frag_node_feats:
            self.frag_node_feats_size += self.rdbe_embedder.num_dim
        if "frag_q" in self.frag_node_feats:
            self.frag_node_feats_size += self.out_degree_embedder.num_dim
        # edges
        self.frag_edge_feats_size = 0
        if "cc" in self.frag_edge_feats:
            self.frag_edge_feats_size += mol_pool_out_size
        if "base_formula" in self.frag_edge_feats:
            self.frag_edge_feats_size += self.num_elements * self.formula_embedder.num_dim
        if "complement" in self.frag_edge_feats:
            self.frag_edge_feats_size += self.complement_embedder.num_dim

    def _apply_ion_channel_factorized(
        self,
        logits: th.Tensor,
        frag_embed: th.Tensor,
        spec_prec_type: th.Tensor | None = None,
        frag_node_batch_idxs: th.Tensor | None = None,
    ) -> th.Tensor:
        """Add a low-rank ion-channel residual to per-node H/CMF logits.

        Two mutually exclusive modes:
        - ``ion_channel_factorized``: ``residual = node_proj(x) @ channel_embed.T + bias``,
          a static node x channel term.
        - ``ion_channel_adduct_gated``: ``residual = (node_proj(x) * gate[adduct]) @
          channel_embed.T``, a node x channel x adduct trilinear term with no bias.

        Args:
            logits: Per-node channel logits of shape (num_nodes, mlp_output_dim).
            frag_embed: Per-node fragment embedding of shape (num_nodes, mlp_input_dim).
            spec_prec_type: Per-sample adduct indices of shape (batch_size,). Required for the
                adduct-gated mode.
            frag_node_batch_idxs: Batch index per node of shape (num_nodes,). Required for the
                adduct-gated mode.

        Returns:
            Logits of shape (num_nodes, mlp_output_dim) with the residual added.
        """
        if not (self.ion_channel_factorized or self.ion_channel_adduct_gated):
            return logits
        node_factor = self.ion_channel_node_proj(frag_embed)
        if self.ion_channel_adduct_gated:
            if spec_prec_type is None or frag_node_batch_idxs is None:
                raise ValueError(
                    "ion_channel_adduct_gated requires spec_prec_type and frag_node_batch_idxs"
                )
            gate = self.ion_channel_adduct_gate(spec_prec_type)[frag_node_batch_idxs]
            node_factor = node_factor * gate
        residual = node_factor @ self.ion_channel_embed.weight.T
        if self.ion_channel_bias is not None:
            residual = residual + self.ion_channel_bias
        return logits + residual

    def _candidate_ion_features(
        self,
        frag_embed: th.Tensor,
        frag_x: th.Tensor,
        frag_node_feat_idxs: th.Tensor,
        frag_node_batch_idxs: th.Tensor,
        frag_joint_mask: th.Tensor,
        batch_size: int,
        node_start: int = 0,
        node_end: int | None = None,
    ) -> th.Tensor:
        """Build per-(node, H-channel) feature rows for the observable ion scorer."""
        device = frag_embed.device
        H = self.mlp_output_dim
        total_nodes = frag_embed.shape[0]
        if node_end is None:
            node_end = total_nodes
        frag_embed_chunk = frag_embed[node_start:node_end]
        N = frag_embed_chunk.shape[0]
        h_pos = th.arange(H, device=device)
        h_pos_2d = h_pos.unsqueeze(0).expand(N, H)
        candidate_parts = [
            self.candidate_h_channel_embedder(h_pos_2d.reshape(-1, 1)),
        ]

        crf_width = 2 * self.num_hs + 1
        cmf_flag = (h_pos_2d >= crf_width).long()
        candidate_parts.append(self.candidate_cmf_flag_embedder(cmf_flag.reshape(-1, 1)))

        if self.candidate_ion_formula_features:
            base_formula = get_node_feats(frag_x, frag_node_feat_idxs, "base_formula").reshape(
                total_nodes, -1
            )
            node_h_counts = get_node_feats(frag_x, frag_node_feat_idxs, "h_counts").reshape(
                total_nodes, -1
            )
            base_formula_chunk = base_formula[node_start:node_end]
            node_h_counts_chunk = node_h_counts[node_start:node_end]
            h_lookup = h_pos.remainder(crf_width).clamp(max=node_h_counts.shape[1] - 1)
            candidate_h = node_h_counts_chunk[:, h_lookup].clamp(min=0)
            candidate_formula = base_formula_chunk.unsqueeze(1).expand(N, H, -1).clone()
            candidate_formula[:, :, self.num_elements - 1] = candidate_h
            candidate_formula = candidate_formula.reshape(N * H, self.num_elements)
            candidate_parts.append(self.formula_embedder(candidate_formula))

            if self.candidate_ion_neutral_loss_features:
                depth_raw = get_node_feats(frag_x, frag_node_feat_idxs, "depth").reshape(
                    total_nodes, -1
                )
                root_mask = depth_raw[:, 0].bool()
                precursor_formula = th.zeros(
                    batch_size,
                    self.num_elements,
                    dtype=base_formula.dtype,
                    device=device,
                )
                root_formula = base_formula[root_mask].clone()
                root_h_counts = node_h_counts[root_mask, 0].clamp(min=0)
                root_formula[:, self.num_elements - 1] = root_h_counts
                precursor_formula[frag_node_batch_idxs[root_mask]] = root_formula
                frag_node_batch_chunk = frag_node_batch_idxs[node_start:node_end]
                neutral_loss_formula = (
                    precursor_formula[frag_node_batch_chunk]
                    .unsqueeze(1)
                    .expand(N, H, -1)
                    - candidate_formula.reshape(N, H, self.num_elements)
                )
                candidate_parts.append(
                    self.formula_embedder(
                        neutral_loss_formula.reshape(N * H, self.num_elements)
                    )
                )

        candidate_feats = th.cat(candidate_parts, dim=1)
        joint_mask_chunk = frag_joint_mask.reshape(total_nodes, H)[node_start:node_end]
        candidate_feats = candidate_feats * joint_mask_chunk.reshape(-1, 1).to(
            candidate_feats.dtype
        )
        frag_embed_expanded = (
            frag_embed_chunk.unsqueeze(1)
            .expand(N, H, frag_embed_chunk.shape[1])
            .reshape(N * H, -1)
        )
        return th.cat([frag_embed_expanded, candidate_feats], dim=1)

    def _candidate_ion_logits(
        self,
        frag_embed: th.Tensor,
        frag_x: th.Tensor,
        frag_node_feat_idxs: th.Tensor,
        frag_node_batch_idxs: th.Tensor,
        frag_joint_mask: th.Tensor,
        batch_size: int,
    ) -> th.Tensor:
        """Score candidate rows in chunks to avoid materializing all N*H rows at once."""
        H = self.mlp_output_dim
        total_nodes = frag_embed.shape[0]
        chunk_rows = self.candidate_ion_chunk_size
        if chunk_rows <= 0:
            chunk_rows = total_nodes * H
        node_step = max(1, chunk_rows // H)
        logits = []
        for node_start in range(0, total_nodes, node_step):
            node_end = min(total_nodes, node_start + node_step)

            def _chunk_forward(
                _frag_embed: th.Tensor,
                _frag_x: th.Tensor,
                _frag_node_feat_idxs: th.Tensor,
                _frag_node_batch_idxs: th.Tensor,
                _frag_joint_mask: th.Tensor,
                _node_start: int = node_start,
                _node_end: int = node_end,
            ) -> th.Tensor:
                candidate_input = self._candidate_ion_features(
                    frag_embed=_frag_embed,
                    frag_x=_frag_x,
                    frag_node_feat_idxs=_frag_node_feat_idxs,
                    frag_node_batch_idxs=_frag_node_batch_idxs,
                    frag_joint_mask=_frag_joint_mask,
                    batch_size=batch_size,
                    node_start=_node_start,
                    node_end=_node_end,
                )
                return self.formula_module(candidate_input).reshape(_node_end - _node_start, H)

            if self.training and self.candidate_ion_checkpoint:
                chunk_logits = th_checkpoint.checkpoint(
                    _chunk_forward,
                    frag_embed,
                    frag_x,
                    frag_node_feat_idxs,
                    frag_node_batch_idxs,
                    frag_joint_mask,
                    use_reentrant=False,
                )
            else:
                chunk_logits = _chunk_forward(
                    frag_embed,
                    frag_x,
                    frag_node_feat_idxs,
                    frag_node_batch_idxs,
                    frag_joint_mask,
                )
            logits.append(chunk_logits)
        return th.cat(logits, dim=0)

    def get_compile(self, **kwargs):
        if check_pyg_full_compile():
            return th.compile(self, **kwargs)
        else:
            self.compile_submodules(**kwargs)
            return self

    def compile_submodules(self, **kwargs):
        """pyg does not support dynamic shape compiling"""
        self.formula_embedder = th.compile(self.formula_embedder, **kwargs)
        self.depth_embedder = th.compile(self.depth_embedder, **kwargs)
        self.complement_embedder = th.compile(self.complement_embedder, **kwargs)
        self.out_degree_embedder = th.compile(self.out_degree_embedder, **kwargs)
        self.in_degree_embedder = th.compile(self.in_degree_embedder, **kwargs)
        self.num_heavy_atoms_embedder = th.compile(self.num_heavy_atoms_embedder, **kwargs)
        if hasattr(self, "num_cut_bonds_embedder"):
            self.num_cut_bonds_embedder = th.compile(self.num_cut_bonds_embedder, **kwargs)
        if hasattr(self, "rdbe_embedder"):
            self.rdbe_embedder = th.compile(self.rdbe_embedder, **kwargs)
        if (
            hasattr(self, "candidate_h_channel_embedder")
            and self.candidate_h_channel_embedder is not None
        ):
            self.candidate_h_channel_embedder = th.compile(
                self.candidate_h_channel_embedder, **kwargs
            )
        if (
            hasattr(self, "candidate_cmf_flag_embedder")
            and self.candidate_cmf_flag_embedder is not None
        ):
            self.candidate_cmf_flag_embedder = th.compile(
                self.candidate_cmf_flag_embedder, **kwargs
            )
        if hasattr(self, "ce_embedder"):
            self.ce_embedder = th.compile(self.ce_embedder, **kwargs)
        if hasattr(self, "m_ce_embedder"):
            self.m_ce_embedder = th.compile(self.m_ce_embedder, **kwargs)
        if hasattr(self, "boundary_pair_proj"):
            self.boundary_pair_proj = th.compile(self.boundary_pair_proj, **kwargs)
        if hasattr(self, "cut_bond_encoder"):
            self.cut_bond_encoder = th.compile(self.cut_bond_encoder, **kwargs)
        if hasattr(self, "frag_q_proj"):
            self.frag_q_proj = th.compile(self.frag_q_proj, **kwargs)
        if check_pyg_compile():
            self.mol_embedder = pyg.compile(self.mol_embedder, **kwargs)  # type: ignore[assignment]
            self.frag_embedder = pyg.compile(self.frag_embedder, **kwargs)  # type: ignore[assignment]

    def forward(
        self,
        mol_pyg: pyg.data.Data,
        frag_pyg: pyg.data.Data,
        mol_num_nodes: th.Tensor,
        frag_num_nodes: th.Tensor,
        frag_formula_peak_idxs: th.Tensor,
        frag_formula_peak_mzs: th.Tensor,
        frag_formula_peak_probs: th.Tensor,
        frag_formula_sizes: th.Tensor,
        frag_formula_cumsizes: th.Tensor,
        frag_formula_peak_sizes: th.Tensor,
        frag_formula_str: np.ndarray | None = None,
        spec_ce: th.Tensor | None = None,
        spec_ce_batch_idxs: th.Tensor | None = None,
        spec_prec_type: th.Tensor | None = None,
        spec_inst_type: th.Tensor | None = None,
        spec_frag_mode: th.Tensor | None = None,
        spec_prec_type_str: np.ndarray | None = None,
        return_explainability_tensors: bool = False,
        **kwargs,
    ):
        """forward methods for joint predictor

        Args:
            mol_pyg (pyg.data.Data): molecule pyg data object
            frag_pyg (pyg.data.Data): fragmentation graph pyg data object
            mol_num_nodes (th.Tensor): number of nodes in molecule graph
            frag_num_nodes (th.Tensor): number of nodes in fragmentation graph
            frag_formula_peak_idxs (th.Tensor): _description_
            frag_formula_peak_mzs (th.Tensor): _description_
            frag_formula_peak_probs (th.Tensor): _description_
            frag_formula_sizes (th.Tensor): _description_
            frag_formula_cumsizes (th.Tensor): _description_
            frag_formula_peak_sizes (th.Tensor): _description_

        Returns:
            _type_: _description_
        """

        # Fix: Explicitly cast PyG batch tensors for autocast compatibility
        # if th.is_autocast_enabled():
        #   mol_pyg.x = mol_pyg.x.to(th.bfloat16)
        #   if mol_pyg.edge_attr is not None:
        #       mol_pyg.edge_attr = mol_pyg.edge_attr.to(th.bfloat16)
        #   frag_pyg.x = frag_pyg.x.to(th.bfloat16)
        #   if frag_pyg.edge_attr is not None:
        #       frag_pyg.edge_attr = frag_pyg.edge_attr.to(th.bfloat16)

        # mol_x: mol level node feature matrix
        # mol_edge_index: mol graph connectivity in COO format with shape [2, num_edges]
        # edge_attr: mol graph edge feature matrix with shape [num_edges, num_edge_features]
        # batch: sample idx repsect to current batch
        mol_x, mol_edge_index, mol_edge_attr, mol_batch = (
            mol_pyg.x,
            mol_pyg.edge_index,
            mol_pyg.edge_attr,
            mol_pyg.batch,
        )

        # frag_x: frag-graph level node feature matrix
        # frag_edge_index: frag graph connectivity in COO format with shape [2, num_edges]
        # frag_edge_attr: frag graph edge feature matrix with shape [num_edges, num_edge_features]
        # batch: sample idx repsect to current batch
        frag_x, frag_edge_index, frag_edge_attr, frag_batch = (
            frag_pyg.x,
            frag_pyg.edge_index,
            frag_pyg.edge_attr,
            frag_pyg.batch,
        )

        # Type assertions for batch indices (PyG types them as Any | None)
        assert mol_batch is not None, "mol_batch must not be None"
        assert frag_batch is not None, "frag_batch must not be None"

        device = mol_num_nodes.device
        # int_dtype = mol_edge_index.dtype
        float_dtype = mol_edge_attr.dtype
        frag_node_feat_idxs = frag_pyg.node_feat_idxs[0]
        frag_edge_feat_idxs = frag_pyg.edge_feat_idxs[0]
        batch_frag_num_nodes = frag_x.shape[0]
        batch_frag_num_edges = frag_edge_index.shape[1]
        batch_frag_num_formulae = frag_formula_cumsizes[-1]
        batch_size = frag_batch[-1] + 1

        # get ce value
        if self.use_ace:
            ace = kwargs["spec_ace"]
            ace_batch_idxs = kwargs["spec_ace_batch_idxs"]
        else:
            ace = None
            ace_batch_idxs = None
        ce_embed = self.embed_ce(
            spec_ce, spec_ce_batch_idxs, batch_size, ace=ace, ace_batch_idxs=ace_batch_idxs
        )
        # get prec value
        prec_embed = self.embed_prec(spec_prec_type)
        # get inst value
        inst_embed = self.embed_inst(spec_inst_type)
        # get frag_mode value
        frag_mode_embed = self.embed_frag_mode(spec_frag_mode)

        # Combined context vector (when enabled): consolidate the active metadata embeddings into
        # one vector, in the same fixed order used to size context_embedder's input dim.
        context_embed = None
        if self.context_embedding:
            _ctx_embeds = [
                e for e in (ce_embed, prec_embed, inst_embed, frag_mode_embed) if e is not None
            ]
            context_embed = self.context_embedder(_ctx_embeds)

        # Type assertions for embedding tensors (narrow types from Any | None to Tensor)

        # Precompute per-mol node counts once from the cumulative prefix sum (avoids up to 4
        # redundant th.unique scans over mol_batch when multiple embeddings insert at "mol").
        mol_batch_counts = mol_num_nodes[1:] - mol_num_nodes[:-1]

        _mol_extras = []
        if self.context_embedding:
            # context subsumes the individual metadata; inject only the context vector
            if "mol" in self.context_insert_locations:
                _mol_extras.append(context_embed)
        else:
            if "mol" in self.ce_insert_locations:
                assert ce_embed is not None, "ce_embed must not be None"
                _mol_extras.append(ce_embed)
            if self.prec_insert_location == "mol":
                assert prec_embed is not None, "prec_embed must not be None"
                _mol_extras.append(prec_embed)
            if self.inst_insert_location == "mol":
                assert inst_embed is not None, "inst_embed must not be None"
                _mol_extras.append(inst_embed)
            if self.frag_mode_insert_location == "mol":
                assert frag_mode_embed is not None, "frag_mode_embed must not be None"
                _mol_extras.append(frag_mode_embed)
        if _mol_extras:
            mol_x = th.cat(
                [mol_x, th.repeat_interleave(th.cat(_mol_extras, dim=1), mol_batch_counts, dim=0)],
                dim=1,
            )

        # get per-atom embeddings
        mol_embed_gnn = self.mol_embedder(
            mol_x,
            mol_batch,
            mol_edge_index,
            mol_edge_attr,
            ctx=context_embed if self.mol_msg_gate else None,
        )
        if self.mol_context_input_only:
            mol_embed_gnn = self.mol_embedder.input_project(mol_x)

        # pool mol embeddings
        # For sum/mean, use segment_csr (ptr-only path) — same logic as frag_pool below.
        # mol_num_nodes is already the CSR ptr (cumulative atom counts, shape [B+1]).
        # Max is excluded: segment_csr max allocates ~3× peak memory vs scatter_max.
        if self.mol_pool_type in ("sum", "mean"):
            mol_embed_gnn_pool = self.mol_pool(
                mol_embed_gnn, index=None, ptr=mol_num_nodes, dim_size=batch_size
            )
        else:
            mol_embed_gnn_pool = self.mol_pool(mol_embed_gnn, mol_batch)

        # process dag
        # create interstage
        frag_ndata, frag_edata = [], []
        # node atom embeddings
        # frag_node_mask is needed by cc, boundary_cc, and boundary_pair features
        _needs_frag_node_mask = any(
            f in self.frag_node_feats for f in ("cc", "boundary_cc", "boundary_pair")
        )
        if _needs_frag_node_mask:
            frag_node_mask = th_long_to_mask(
                get_node_feats(frag_x, frag_node_feat_idxs, "cc").to(device)
            )
        if "cc" in self.frag_node_feats:
            cc_mask = frag_node_mask
            if self.cc_corruption_frac > 0.0:
                flip = th.rand_like(cc_mask, dtype=float_dtype) < self.cc_corruption_frac
                cc_mask = cc_mask ^ flip
            if self.training and self.cc_atom_dropout > 0.0:
                # per-atom dropout: independently drop atoms from each fragment's membership set
                drop = th.rand_like(cc_mask, dtype=float_dtype) < self.cc_atom_dropout
                cc_mask = cc_mask & ~drop
                # never empty a fragment (an empty pool would yield a degenerate cc embedding):
                # restore the true mask for any fragment that lost every atom
                _emptied = ~cc_mask.any(dim=1)
                if _emptied.any():
                    cc_mask = th.where(_emptied.unsqueeze(1), frag_node_mask, cc_mask)
            frag_node_mask_idxs = th.nonzero(cc_mask).long()
            # Compute offset to convert local atom indices to global batch indices
            frag_node_offsets = mol_num_nodes[
                th.bucketize(frag_node_mask_idxs[:, 0], frag_num_nodes, right=True) - 1
            ]
            frag_node_mask_idxs[:, 1] = frag_node_mask_idxs[:, 1] + frag_node_offsets

            # Validate frag_node_mask_idxs are within bounds
            valid_frag_mask = frag_node_mask_idxs[:, 1] < mol_embed_gnn.shape[0]
            if not th.all(valid_frag_mask):
                logger.warning(
                    f"Filtered {(~valid_frag_mask).sum()} invalid frag node indices out of {len(frag_node_mask_idxs)}"
                )
                frag_node_mask_idxs = frag_node_mask_idxs[valid_frag_mask]

            # Aggregate CC atom embeddings per fragment using pooling
            if frag_node_mask_idxs.shape[0] == 0:
                frag_node_mask_embed = th.zeros(
                    batch_frag_num_nodes, mol_embed_gnn.shape[1], device=device, dtype=float_dtype
                )
            elif self.mol_pool_type in ("sum", "mean"):
                M_cc = th.sparse_coo_tensor(
                    th.stack([frag_node_mask_idxs[:, 0], frag_node_mask_idxs[:, 1]]),
                    th.ones(frag_node_mask_idxs.shape[0], device=device, dtype=float_dtype),
                    (batch_frag_num_nodes, mol_embed_gnn.shape[0]),
                )
                frag_node_mask_embed = th.sparse.mm(M_cc, mol_embed_gnn)
                if self.mol_pool_type == "mean":
                    degree = (
                        th.bincount(frag_node_mask_idxs[:, 0], minlength=batch_frag_num_nodes)
                        .to(float_dtype)
                        .clamp(min=1)
                        .unsqueeze(1)
                    )
                    frag_node_mask_embed = frag_node_mask_embed / degree
            else:
                frag_node_embed = th.index_select(mol_embed_gnn, 0, frag_node_mask_idxs[:, 1])
                frag_node_mask_embed = self.mol_pool(
                    frag_node_embed, frag_node_mask_idxs[:, 0], dim_size=batch_frag_num_nodes
                )

            # frag_node_mask_embed can be one of following:
            # 1. add/sub: Simple arithmetic operations
            # 2. linear: Linear projection of concatenation
            # 3. gated: Learnable gate to mix embeddings
            # 4. mlp: Deep nonlinear fusion
            # 5. bilinear: Bilinear interaction
            # 6. direct: No fusion, use fragment embedding only

            if self.cc_interstage_use_rest:
                # compute rest node idxs (non-CC atoms per fragment)
                # NOTE: ~frag_node_mask includes padding zeros (bits beyond mol size) so
                # bounds-filtering is required to exclude them.
                rest_node_mask_idx = th.nonzero(~frag_node_mask).long()
                rest_node_offsets = mol_num_nodes[
                    th.bucketize(rest_node_mask_idx[:, 0], frag_num_nodes, right=True) - 1
                ]
                rest_node_mask_idx[:, 1] = rest_node_mask_idx[:, 1] + rest_node_offsets
                valid_rest_mask = rest_node_mask_idx[:, 1] < mol_embed_gnn.shape[0]
                rest_node_mask_idx = rest_node_mask_idx[valid_rest_mask]
                rest_node_embed = th.index_select(mol_embed_gnn, 0, rest_node_mask_idx[:, 1])
                mol_pool_embed = self.mol_pool(rest_node_embed, rest_node_mask_idx[:, 0])
            else:
                mol_pool_embed = mol_embed_gnn_pool[frag_batch]

            if self.cc_interstage_type == "add":
                frag_node_mask_embed = frag_node_mask_embed + mol_pool_embed
            elif self.cc_interstage_type == "sub":
                frag_node_mask_embed = frag_node_mask_embed - mol_pool_embed
            elif self.cc_interstage_type == "linear":
                frag_node_mask_embed = self.cc_interstage(
                    th.cat([frag_node_mask_embed, mol_pool_embed], dim=1)
                )
            elif self.cc_interstage_type == "gated":
                # Gate controls mixing: g * frag + (1-g) * mol
                gate = self.cc_interstage(th.cat([frag_node_mask_embed, mol_pool_embed], dim=1))
                frag_node_mask_embed = gate * frag_node_mask_embed + (1 - gate) * mol_pool_embed
            elif self.cc_interstage_type == "mlp":
                # MLPBlocks fusion (handles residuals internally if use_residuals=True)
                frag_node_mask_embed = self.cc_interstage(
                    th.cat([frag_node_mask_embed, mol_pool_embed], dim=1)
                )
            elif self.cc_interstage_type == "bilinear":
                # Bilinear interaction: captures pairwise interactions
                interaction = self.cc_interstage_bilinear(frag_node_mask_embed, mol_pool_embed)
                frag_node_mask_embed = frag_node_mask_embed + self.cc_interstage_proj(interaction)
            else:
                assert self.cc_interstage_type == "direct", self.cc_interstage_type
                frag_node_mask_embed = frag_node_mask_embed

            frag_node_mask_embed = self._maybe_apply_cc_channel_dropout(frag_node_mask_embed)
            frag_node_mask_embed = self._maybe_apply_cc_feature_dropout(frag_node_mask_embed)
            frag_ndata.append(frag_node_mask_embed)
        if "boundary_cc" in self.frag_node_feats:
            num_frags_total = frag_x.shape[0]
            boundary_mask = compute_boundary_mask(
                frag_node_mask, mol_edge_index, mol_num_nodes, frag_num_nodes
            )
            boundary_idxs = th.nonzero(boundary_mask).long()
            if boundary_idxs.numel() > 0:
                boundary_offsets = mol_num_nodes[
                    th.bucketize(boundary_idxs[:, 0], frag_num_nodes, right=True) - 1
                ]
                boundary_idxs[:, 1] = boundary_idxs[:, 1] + boundary_offsets
                valid = boundary_idxs[:, 1] < mol_embed_gnn.shape[0]
                boundary_idxs = boundary_idxs[valid]
                if self.mol_pool_type in ("sum", "mean"):
                    M_bcc = th.sparse_coo_tensor(
                        th.stack([boundary_idxs[:, 0], boundary_idxs[:, 1]]),
                        th.ones(boundary_idxs.shape[0], device=device, dtype=float_dtype),
                        (num_frags_total, mol_embed_gnn.shape[0]),
                    )
                    boundary_embed = th.sparse.mm(M_bcc, mol_embed_gnn)
                    if self.mol_pool_type == "mean":
                        degree = (
                            th.bincount(boundary_idxs[:, 0], minlength=num_frags_total)
                            .to(float_dtype)
                            .clamp(min=1)
                            .unsqueeze(1)
                        )
                        boundary_embed = boundary_embed / degree
                else:
                    boundary_embed = th.index_select(mol_embed_gnn, 0, boundary_idxs[:, 1])
                    boundary_embed = self.mol_pool(
                        boundary_embed, boundary_idxs[:, 0], dim_size=num_frags_total
                    )
            else:
                boundary_embed = th.zeros(
                    num_frags_total,
                    mol_embed_gnn.shape[1],
                    device=device,
                    dtype=mol_embed_gnn.dtype,
                )
            frag_ndata.append(boundary_embed)
        if "boundary_pair" in self.frag_node_feats:
            # boundary_pair requires cc to be in frag_node_feats (frag_node_mask must exist)
            # TODO(next version): delete this block when boundary_pair is removed from configs/features.
            num_frags_total = frag_x.shape[0]
            pair_frag_idxs = kwargs["boundary_pair_frag_idxs"]
            pair_in_global = kwargs["boundary_pair_in_idxs"]
            pair_out_global = kwargs["boundary_pair_out_idxs"]
            if pair_frag_idxs.numel() > 0:
                valid = (pair_in_global < mol_embed_gnn.shape[0]) & (
                    pair_out_global < mol_embed_gnn.shape[0]
                )
                pair_frag_idxs = pair_frag_idxs[valid]
                pair_in_global = pair_in_global[valid]
                pair_out_global = pair_out_global[valid]
            if pair_frag_idxs.numel() > 0:
                in_embed = th.index_select(mol_embed_gnn, 0, pair_in_global)
                out_embed = th.index_select(mol_embed_gnn, 0, pair_out_global)
                pair_embed = self.boundary_pair_proj(th.cat([in_embed, out_embed], dim=-1))
                boundary_pair_out = self.mol_pool(
                    pair_embed, pair_frag_idxs, dim_size=num_frags_total
                )
            else:
                boundary_pair_out = th.zeros(
                    num_frags_total,
                    mol_embed_gnn.shape[1],
                    device=device,
                    dtype=mol_embed_gnn.dtype,
                )
            frag_ndata.append(boundary_pair_out)
        if "cut_bond" in self.frag_node_feats:
            # Same cut-bond indices as boundary_pair (no DAG regeneration needed); the only
            # difference is the encoder -- see CutBondEncoder for why the linear one is
            # degenerate. Same mol_pool and output width, so the two are directly comparable.
            num_frags_total = frag_x.shape[0]
            cut_frag_idxs = kwargs["boundary_pair_frag_idxs"]
            cut_in_global = kwargs["boundary_pair_in_idxs"]
            cut_out_global = kwargs["boundary_pair_out_idxs"]
            if cut_frag_idxs.numel() > 0:
                valid = (cut_in_global < mol_embed_gnn.shape[0]) & (
                    cut_out_global < mol_embed_gnn.shape[0]
                )
                cut_frag_idxs = cut_frag_idxs[valid]
                cut_in_global = cut_in_global[valid]
                cut_out_global = cut_out_global[valid]
            if cut_frag_idxs.numel() > 0:
                bond_embed = self.cut_bond_encoder(
                    th.index_select(mol_embed_gnn, 0, cut_in_global),
                    th.index_select(mol_embed_gnn, 0, cut_out_global),
                )
                cut_bond_out = self.mol_pool(bond_embed, cut_frag_idxs, dim_size=num_frags_total)
            else:
                cut_bond_out = th.zeros(
                    num_frags_total,
                    mol_embed_gnn.shape[1],
                    device=device,
                    dtype=mol_embed_gnn.dtype,
                )
            frag_ndata.append(cut_bond_out)
        # edge atom embeddings
        if "cc" in self.frag_edge_feats:
            # connected competents
            frag_edge_mask = th_long_to_mask(
                get_edge_feats(frag_edge_attr, frag_edge_feat_idxs, "cc").to(device)
            )
            frag_edge_mask_idxs = th.nonzero(frag_edge_mask).long()
            frag_edge_node_idxs = frag_edge_index[0][frag_edge_mask_idxs[:, 0]]
            frag_edge_offsets = mol_num_nodes[
                th.bucketize(frag_edge_node_idxs, frag_num_nodes, right=True) - 1
            ]
            frag_edge_mask_idxs[:, 1] = frag_edge_mask_idxs[:, 1] + frag_edge_offsets
            if frag_edge_mask_idxs.shape[0] == 0:
                frag_edge_mask_embed = th.zeros(
                    batch_frag_num_edges, mol_embed_gnn.shape[1], device=device, dtype=float_dtype
                )
            else:
                M_ecc = th.sparse_coo_tensor(
                    th.stack([frag_edge_mask_idxs[:, 0], frag_edge_mask_idxs[:, 1]]),
                    th.ones(frag_edge_mask_idxs.shape[0], device=device, dtype=float_dtype),
                    (batch_frag_num_edges, mol_embed_gnn.shape[0]),
                )
                frag_edge_mask_embed = th.sparse.mm(M_ecc, mol_embed_gnn)
            frag_edata.append(frag_edge_mask_embed)
        # node formulae
        if "base_formula" in self.frag_node_feats:
            frag_base_formula_raw = get_node_feats(
                frag_x, frag_node_feat_idxs, "base_formula"
            ).reshape(batch_frag_num_nodes, -1)
            frag_node_formula = self.formula_embedder(frag_base_formula_raw)
            frag_ndata.append(frag_node_formula)
        # neutral loss formula: precursor_formula - fragment_formula
        # precursor is the root node (depth bit 0 set); broadcast its base_formula to all nodes
        if "neutral_loss_formula" in self.frag_node_feats:
            assert "base_formula" in self.frag_node_feats, (
                "neutral_loss_formula requires base_formula in frag_node_feats"
            )
            assert "depth" in self.frag_node_feats, (
                "neutral_loss_formula requires depth in frag_node_feats"
            )
            depth_raw = get_node_feats(frag_x, frag_node_feat_idxs, "depth").reshape(
                batch_frag_num_nodes, -1
            )
            root_mask = depth_raw[:, 0].bool()
            prec_formula_by_batch = th.zeros(
                batch_size,
                frag_base_formula_raw.shape[1],
                dtype=frag_base_formula_raw.dtype,
                device=device,
            )
            prec_formula_by_batch[frag_batch[root_mask]] = frag_base_formula_raw[root_mask]
            neutral_loss_raw = prec_formula_by_batch[frag_batch] - frag_base_formula_raw
            frag_ndata.append(self.formula_embedder(neutral_loss_raw))
        # edge formulae
        if "base_formula" in self.frag_edge_feats:
            frag_edge_formula = self.formula_embedder(
                get_edge_feats(frag_edge_attr, frag_edge_feat_idxs, "base_formula").reshape(
                    batch_frag_num_edges, -1
                )
            )
            frag_edata.append(frag_edge_formula)
        # node depth
        if "depth" in self.frag_node_feats:
            frag_depth = self.depth_embedder(
                get_node_feats(frag_x, frag_node_feat_idxs, "depth").reshape(
                    batch_frag_num_nodes, -1
                )
            )
            frag_ndata.append(frag_depth)
        # DAG structural scalars: out_degree, in_degree, num_heavy_atoms
        if "out_degree" in self.frag_node_feats:
            frag_ndata.append(
                self.out_degree_embedder(
                    get_node_feats(frag_x, frag_node_feat_idxs, "out_degree").reshape(
                        batch_frag_num_nodes, -1
                    )
                )
            )
        if "in_degree" in self.frag_node_feats:
            frag_ndata.append(
                self.in_degree_embedder(
                    get_node_feats(frag_x, frag_node_feat_idxs, "in_degree").reshape(
                        batch_frag_num_nodes, -1
                    )
                )
            )
        if "num_heavy_atoms" in self.frag_node_feats:
            frag_ndata.append(
                self.num_heavy_atoms_embedder(
                    get_node_feats(frag_x, frag_node_feat_idxs, "num_heavy_atoms").reshape(
                        batch_frag_num_nodes, -1
                    )
                )
            )
        if "num_cut_bonds" in self.frag_node_feats:
            # How many bonds were broken to make this fragment. Not recoverable from anything the
            # model already has: mol_pool is a mean, so boundary_pair / cut_bond discard bond
            # multiplicity, and `depth` is NOT a proxy -- the SMARTS ring-cleavage rules break 4
            # bonds in a single BFS step, so a 4-cut xanthone node and a 1-cut node are both
            # depth 1. Derived from the same precomputed indices boundary_pair uses (batched on
            # every forward regardless of config), so it costs no data and no DAG regeneration.
            # The mol graph is bidirectional (mol_params.pyg_bigraph), so a cut bond yields one
            # entry per direction and BOTH map to the same (frag, inside_atom, outside_atom)
            # triple -- a raw bincount returns exactly 2x the true count. Deduplicate rather than
            # halve, so this stays correct if the mol graph is ever built undirected.
            triples = th.stack(
                [
                    kwargs["boundary_pair_frag_idxs"],
                    kwargs["boundary_pair_in_idxs"],
                    kwargs["boundary_pair_out_idxs"],
                ],
                dim=1,
            )
            uniq_triples = th.unique(triples, dim=0)
            cut_counts = th.bincount(uniq_triples[:, 0], minlength=frag_x.shape[0])[
                : frag_x.shape[0]
            ]
            frag_ndata.append(
                self.num_cut_bonds_embedder(cut_counts.reshape(batch_frag_num_nodes, -1))
            )
        if "rdbe" in self.frag_node_feats:
            # Ring-and-double-bond equivalents: RDBE = 1 + sum_i n_i * (v_i - 2) / 2.
            # base_formula carries NO hydrogen (its H column is identically zero), so H is taken
            # from h_counts column 0 -- the delta-0 (unshifted) H count. RDBE is therefore the
            # SKELETON value: each h-shift moves the true RDBE by +/-0.5, but the h-shift is an
            # OUTPUT dimension of formula_module, not an input, so a per-node feature can only
            # carry the delta-0 value. The shift correction is deterministic downstream.
            # Embedded doubled (2*RDBE) to keep the half-integer resolution of odd-electron
            # species on an integer embedder. Config validity is enforced in __init__.
            h_delta0 = (
                get_node_feats(frag_x, frag_node_feat_idxs, "h_counts")
                .reshape(batch_frag_num_nodes, -1)[:, 0]
                .clamp(min=0)  # -1 marks an unavailable h-state
            )
            counts = frag_base_formula_raw.clone()
            counts[:, self.num_elements - 1] = h_delta0  # H is last in ELEMENTS
            rdbe_x2 = 2 + (counts * self._rdbe_coef_buf).sum(dim=1)  # == 2 * RDBE
            frag_ndata.append(self.rdbe_embedder(rdbe_x2.reshape(batch_frag_num_nodes, -1)))
        if "frag_q" in self.frag_node_feats:
            frag_ndata.append(
                self.frag_q_proj(frag_pyg.frag_q.to(device=device, dtype=float_dtype))
            )
        # edge complement
        if "complement" in self.frag_edge_feats:
            frag_edge_complement = self.complement_embedder(
                get_edge_feats(frag_edge_attr, frag_edge_feat_idxs, "complement").reshape(
                    batch_frag_num_edges, -1
                )
            )
            frag_edata.append(frag_edge_complement)
        # empty feats check
        if len(frag_ndata) == 0:
            assert self.frag_node_feats_size == 0, self.frag_node_feats_size
            frag_ndata.append(th.zeros([batch_frag_num_nodes, 0], dtype=float_dtype, device=device))
        if len(frag_edata) == 0:
            assert self.frag_edge_feats_size == 0, self.frag_edge_feats_size
            frag_edata.append(th.zeros([batch_frag_num_edges, 0], dtype=float_dtype, device=device))

        # charge migration fragmentation flag check
        include_cmf = "cmf_h_formulae_idx" in self.frag_node_feats
        # get output formula aggregation
        frag_node_batch_idxs = frag_batch
        frag_formula_batch_idxs = th.repeat_interleave(
            th.arange(batch_size, device=device),
            frag_formula_sizes,  # equivalent to cumsizes[1:] - cumsizes[:-1], already computed
        )
        frag_node_offsets = frag_formula_cumsizes[frag_node_batch_idxs]

        h_formulae_idx_feat = get_node_feats(frag_x, frag_node_feat_idxs, "h_formulae_idx")
        frag_joint_formula_idxs = h_formulae_idx_feat + frag_node_offsets.unsqueeze(-1)
        # When CMF features are present the node feature layout may include extra columns
        expected_h_width = 2 * self.num_hs + 1
        if not include_cmf:
            assert h_formulae_idx_feat.shape[1] >= expected_h_width, (
                f"h_formulae_idx feature size {h_formulae_idx_feat.shape[1]} is smaller than expected {expected_h_width} "
                f"(2*num_hs+1); DAG was built with fewer h-transfers than num_hs={self.num_hs}"
            )
        else:
            # allow extra columns when CMF is included but ensure minimum expected width
            if h_formulae_idx_feat.shape[1] < expected_h_width:
                raise AssertionError(
                    f"h_formulae_idx feature size {h_formulae_idx_feat.shape[1]} smaller than expected {expected_h_width} (2*num_hs+1) even with CMF enabled"
                )
        if include_cmf:
            # if we use cmf, we need to extend the formula indices
            frag_joint_formula_idxs = th.cat(
                [
                    frag_joint_formula_idxs,
                    get_node_feats(frag_x, frag_node_feat_idxs, "cmf_h_formulae_idx")
                    + frag_node_offsets.unsqueeze(-1),
                ],
                dim=1,
            )

        # Trim h-columns if the DAG was built with more h-transfers than model's num_hs.
        # This allows loading h4 DAGs while training/inferring with num_hs=3 (or lower).
        # Layout: [Δ0, Δ-1, Δ+1, Δ-2, Δ+2, ..., Δ-num_hs, Δ+num_hs] — trimming from
        # the end removes the largest-magnitude shifts first.
        frag_joint_formula_idxs = frag_joint_formula_idxs.reshape(batch_frag_num_nodes, -1)
        h_width = frag_joint_formula_idxs.shape[1] // (2 if include_cmf else 1)
        num_hs_diff = (h_width - 1) // 2 - self.num_hs
        assert num_hs_diff >= 0, (
            f"DAG max_h_transfer ({(h_width - 1) // 2}) < model num_hs ({self.num_hs}); "
            "regenerate DAGs with at least num_hs h-transfers"
        )
        if num_hs_diff > 0 and include_cmf:
            raise NotImplementedError(
                "Trimming h-shifts on CMF DAGs is not yet supported. "
                "Regenerate DAGs with max_h_transfer matching num_hs."
            )
        if num_hs_diff > 0:
            # th.unique deferred here — only needed when DAG max_h_transfer > model num_hs
            frag_formula_idxs_pretrim = th.unique(frag_joint_formula_idxs)
            # Remove last 2*num_hs_diff columns (drops Δ±(num_hs+1) … Δ±dag_max_h)
            frag_joint_formula_idxs = frag_joint_formula_idxs[:, : -2 * num_hs_diff].flatten()

            # Find unique formula indices still referenced after trimming (+ NULL per batch)
            frag_joint_formula_idxs_un, frag_joint_formula_idxs_inv = th.unique(
                th.cat([frag_joint_formula_idxs, frag_formula_cumsizes[:-1]], dim=0),
                return_inverse=True,
            )
            frag_joint_formula_idxs_inv = frag_joint_formula_idxs_inv[
                : frag_joint_formula_idxs.shape[0]
            ]

            # Mask for which pre-trim peaks survive into the trimmed set
            frag_formula_peak_mask = th.isin(
                frag_formula_idxs_pretrim[
                    ~th.isin(frag_formula_idxs_pretrim, frag_formula_cumsizes[:-1])
                ],
                frag_joint_formula_idxs_un,
            )

            # Update formula counts and batch mapping
            batch_frag_num_formulae = frag_joint_formula_idxs_un.shape[0]
            frag_formula_batch_idxs = frag_formula_batch_idxs[frag_joint_formula_idxs_un]
            frag_joint_formula_idxs = th.arange(batch_frag_num_formulae, device=device)[
                frag_joint_formula_idxs_inv
            ]

            frag_formula_idxs = th.arange(batch_frag_num_formulae, device=device)
            frag_formula_sizes = scatter_reduce(
                th.ones_like(frag_formula_batch_idxs),
                frag_formula_batch_idxs,
                reduce="sum",
                dim_size=batch_size,
            )
            assert not th.any(frag_formula_sizes <= 1), frag_formula_sizes

            frag_formula_cumsizes = th.cumsum(
                th.cat(
                    [
                        th.zeros([1], device=device, dtype=frag_formula_sizes.dtype),
                        frag_formula_sizes,
                    ],
                    dim=0,
                ),
                dim=0,
            )
            frag_node_offsets = frag_formula_cumsizes[frag_node_batch_idxs]

            # Update peak-related tensors
            non_null_mask = ~th.isin(frag_formula_idxs, frag_formula_cumsizes[:-1])
            frag_formula_peak_idxs = frag_formula_idxs[non_null_mask]
            frag_formula_peak_idxs = (
                frag_formula_peak_idxs
                - frag_formula_cumsizes[:-1][frag_formula_batch_idxs[non_null_mask]]
            )
            frag_formula_peak_probs = frag_formula_peak_probs[frag_formula_peak_mask]
            frag_formula_peak_mzs = frag_formula_peak_mzs[frag_formula_peak_mask]
            frag_formula_peak_sizes = frag_formula_sizes - 1
        else:
            frag_joint_formula_idxs = frag_joint_formula_idxs.flatten()
        # get isomorphism aggregation
        if self.nb_iso:
            frag_nb_idxs = get_node_feats(frag_x, frag_node_feat_idxs, "nb_iso_idx").flatten()
            frag_nb_offsets = scatter_reduce(
                frag_nb_idxs, frag_node_batch_idxs, reduce="amax", dim_size=batch_size
            )
            frag_nb_offsets = th.cat(
                [
                    th.zeros([1], dtype=frag_nb_offsets.dtype, device=device),
                    frag_nb_offsets + 1,
                ],
                dim=0,
            )
            frag_nb_offsets = th.cumsum(frag_nb_offsets, dim=0)
            batch_frag_nb_num_nodes = frag_nb_offsets[-1].item()
            frag_nb_offsets = th.gather(
                input=frag_nb_offsets[:-1], index=frag_node_batch_idxs, dim=0
            )
            frag_nb_idxs = frag_nb_idxs + frag_nb_offsets
            assert th.max(frag_nb_idxs) < batch_frag_nb_num_nodes, (
                th.max(frag_nb_idxs),
                batch_frag_nb_num_nodes,
            )
            frag_nb_un_idxs, frag_nb_inv_idxs = th.unique(frag_nb_idxs, return_inverse=True)

        # assemble all features for dag
        # concatenate everything
        frag_x_embed = th.cat(frag_ndata, dim=-1)
        # concatenate everything
        frag_edge_attr_embed = th.cat(frag_edata, dim=-1)

        # define frag network
        frag_embed_gnn = self.frag_embedder(
            frag_x_embed, frag_node_batch_idxs, frag_edge_index, frag_edge_attr_embed
        )
        frag_embed_node = self.frag_embedder.input_project(frag_x_embed)
        _need_gnn_pool = self.frag_pool_combine != "none" or self.predict_oos
        # For sum/mean, precompute CSR ptr and call frag_pool with index=None so PyG routes
        # through segment_csr instead of scatter. segment_csr requires index=None (or
        # _deterministic=True) to activate; passing both index and ptr keeps scatter active.
        # Benefits: contiguous memory reads, ptr is (B+1,) vs scatter index (N,); mean also
        # saves ~50% peak memory (no internal count buffer). frag_batch is sorted in PyG
        # batches so ptr is always valid.
        # NOTE: "max" is intentionally excluded — segment_csr for max allocates ~3× the peak
        # memory of scatter_max (temporary workspace for backward), which outweighs the 8–9×
        # speed gain when memory is the primary constraint.
        if self.frag_pool_type in ("sum", "mean") and _need_gnn_pool:
            _frag_counts = th.bincount(frag_batch, minlength=batch_size)
            frag_ptr = th.cat([_frag_counts.new_zeros(1), _frag_counts.cumsum(0)])
        else:
            frag_ptr = None
        # _pool_idx: pass None when using ptr (segment_csr path), frag_batch otherwise.
        _pool_idx = None if frag_ptr is not None else frag_batch
        frag_embed_gnn_pool = (
            self.frag_pool(frag_embed_gnn, _pool_idx, ptr=frag_ptr, dim_size=batch_size)
            if _need_gnn_pool
            else None
        )
        if self.frag_pool_combine == "subtract":
            frag_embed_node_pool = self.frag_pool(
                frag_embed_node, _pool_idx, ptr=frag_ptr, dim_size=batch_size
            )
            frag_embed_gnn = frag_embed_gnn - frag_embed_gnn_pool[frag_batch]
            frag_embed_node = frag_embed_node - frag_embed_node_pool[frag_batch]
        elif self.frag_pool_combine == "add":
            frag_embed_node_pool = self.frag_pool(
                frag_embed_node, _pool_idx, ptr=frag_ptr, dim_size=batch_size
            )
            frag_embed_gnn = frag_embed_gnn + frag_embed_gnn_pool[frag_batch]
            frag_embed_node = frag_embed_node + frag_embed_node_pool[frag_batch]
        elif self.frag_pool_combine == "concat":
            # Global set-context is appended to frag_embed_parts below rather than combined into
            # the per-node embeddings in place.
            pass
        else:
            assert self.frag_pool_combine == "none", self.frag_pool_combine

        # get frag dag embedding
        if self.frag_embed_combine == "cat":
            frag_embed_parts = [frag_embed_gnn, frag_embed_node]
        else:
            assert self.frag_embed_combine == "avg", self.frag_embed_combine
            frag_embed_parts = [0.5 * frag_embed_gnn + 0.5 * frag_embed_node]
        if self.frag_pool_combine == "concat":
            frag_embed_parts.append(frag_embed_gnn_pool[frag_batch])

        # Cache unique batch counts to avoid redundant computation (Memory Opt #1)
        _inject_context_mlp = self.context_embedding and "mlp" in self.context_insert_locations
        _inject_individual_mlp = not self.context_embedding and (
            "mlp" in self.ce_insert_locations
            or self.prec_insert_location == "mlp"
            or self.inst_insert_location == "mlp"
            or self.frag_mode_insert_location == "mlp"
        )
        if _inject_context_mlp or _inject_individual_mlp:
            # frag_num_nodes is a cumulative prefix sum → diff gives per-sample node counts,
            # equivalent to th.unique(frag_node_batch_idxs, return_counts=True)[1] but O(batch)
            unique_batch_counts = frag_num_nodes[1:] - frag_num_nodes[:-1]
            _mlp_extras = []
            if _inject_context_mlp:
                # context subsumes the individual metadata; inject only the context vector
                _mlp_extras.append(context_embed)
            else:
                if "mlp" in self.ce_insert_locations:
                    _mlp_extras.append(ce_embed)
                if self.prec_insert_location == "mlp":
                    _mlp_extras.append(prec_embed)
                if self.inst_insert_location == "mlp":
                    _mlp_extras.append(inst_embed)
                if self.frag_mode_insert_location == "mlp":
                    _mlp_extras.append(frag_mode_embed)
            frag_embed_parts.append(
                th.repeat_interleave(th.cat(_mlp_extras, dim=1), unique_batch_counts, dim=0)
            )

        frag_embed = th.cat(frag_embed_parts, dim=1)
        frag_joint_batch_idxs = th.repeat_interleave(frag_node_batch_idxs, self.mlp_output_dim)
        # Broadcast comparison via reshape — reuses frag_node_offsets (already equals
        # frag_formula_cumsizes[frag_node_batch_idxs], the null-formula index per node) and
        # avoids materializing a full (N×H) null_tiled tensor.
        frag_joint_mask = (
            (
                frag_joint_formula_idxs.reshape(batch_frag_num_nodes, self.mlp_output_dim)
                != frag_node_offsets.unsqueeze(1)
            )
            .float()
            .flatten()
        )

        if self.mlp_output_format == "formula":
            #  predicts joint logits for each (node, formula) pair:
            # log p(f,n)
            if self.candidate_ion_head:
                frag_joint_logits = self._candidate_ion_logits(
                    frag_embed=frag_embed,
                    frag_x=frag_x,
                    frag_node_feat_idxs=frag_node_feat_idxs,
                    frag_node_batch_idxs=frag_node_batch_idxs,
                    frag_joint_mask=frag_joint_mask,
                    batch_size=batch_size,
                )
            else:
                frag_joint_logits = self.formula_module(frag_embed)
            frag_joint_logits = self._apply_ion_channel_factorized(
                frag_joint_logits, frag_embed, spec_prec_type, frag_node_batch_idxs
            )
            # apply mode-conditioned affine scaling if enabled
            if self.frag_mode_scaler is not None:
                assert spec_frag_mode is not None, "spec_frag_mode required for frag_mode_scaler"
                frag_joint_logits = self.frag_mode_scaler(
                    frag_joint_logits, spec_frag_mode, frag_node_batch_idxs
                )
            # apply CE-conditioned FiLM scaling if enabled
            if self.ce_scaler is not None:
                frag_joint_logits = self.ce_scaler(
                    frag_joint_logits,
                    frag_node_batch_idxs,
                    nce_stats=kwargs.get("spec_nce_stats"),
                    ace_stats=kwargs.get("spec_ace_stats"),
                )
            # apply adduct-type FiLM scaling if enabled
            if self.adduct_scaler is not None:
                assert spec_prec_type is not None, "spec_prec_type required for adduct_scaler"
                frag_joint_logits = self.adduct_scaler(
                    frag_joint_logits, spec_prec_type, frag_node_batch_idxs
                )
            # apply combined stratum FiLM scaling if enabled
            if self.stratum_scaler is not None:
                assert spec_inst_type is not None, "spec_inst_type required for stratum_scaler"
                assert spec_frag_mode is not None, "spec_frag_mode required for stratum_scaler"
                assert spec_prec_type is not None, "spec_prec_type required for stratum_scaler"
                frag_joint_logits = self.stratum_scaler(
                    frag_joint_logits,
                    spec_inst_type,
                    spec_frag_mode,
                    spec_prec_type,
                    frag_node_batch_idxs,
                )
            # apply CE × depth prior if enabled (§3c)
            if self.depth_ce_bias_layer is not None:
                assert ce_embed is not None, "depth_ce_bias requires CE embedding"
                depth_raw = get_node_feats(frag_x, frag_node_feat_idxs, "depth").reshape(
                    batch_frag_num_nodes, -1
                )
                depth_idx = depth_raw.argmax(dim=1).clamp(0, self.num_depth)  # (n_nodes,)
                bias_per_depth = self.depth_ce_bias_layer(ce_embed)            # (batch, num_depth+1)
                per_node_bias = bias_per_depth[frag_node_batch_idxs, depth_idx]  # (n_nodes,)
                frag_joint_logits = frag_joint_logits + per_node_bias.unsqueeze(-1)
            if include_cmf:
                # CMF Δh≠0 channels (columns crf_width+1..) that map to NULL should not
                # consume probability mass before softmax.  This is always the case for
                # [M+H]+/[M-H]- (direct-charge: only Δh=0 is a real CMF peak) and also
                # applies per-node for any adduct where a Δh≠0 shift was not generated.
                crf_width = 2 * self.num_hs + 1
                formula_idxs_2d = frag_joint_formula_idxs.reshape(
                    batch_frag_num_nodes, self.mlp_output_dim
                )
                cmf_dhn0_null = (
                    formula_idxs_2d[:, crf_width + 1 :] == frag_node_offsets.unsqueeze(1)
                )
                if cmf_dhn0_null.any():
                    null_mask = F.pad(cmf_dhn0_null, (crf_width + 1, 0))
                    frag_joint_logits = frag_joint_logits.masked_fill(
                        null_mask, LOG_ZERO(frag_joint_logits.dtype)
                    )
            assert frag_joint_logits.shape[1] == self.mlp_output_dim, (
                frag_joint_logits.shape[1],
                2 * self.num_hs + 1,
            )
            # print("> frag_joint_logits shape:", frag_joint_logits.shape)
            frag_joint_logits = frag_joint_logits.flatten()

            # compute all joint log probabilits
            frag_joint_logprobs = scatter_logsoftmax(frag_joint_logits, frag_joint_batch_idxs)
            # compute total NULL probability (before renormalization)
            frag_null_formula_logprob = scatter_logsumexp(
                (1.0 - frag_joint_mask) * frag_joint_logprobs
                + frag_joint_mask * LOG_ZERO(frag_joint_logprobs.dtype),
                frag_joint_batch_idxs,
            )
            frag_null_formula_logprob = th.clamp(frag_null_formula_logprob, max=0.0)

            if self.mask_null_formula:
                # compute non-NULL renormalized intensity
                frag_joint_logprobs = scatter_masked_softmax(
                    frag_joint_logits, frag_joint_mask, frag_joint_batch_idxs, log=True
                )

            # reshape
            # Each row corresponds to a node, and each column to a possible formula for that node.
            frag_joint_logprobs = frag_joint_logprobs.reshape(-1, self.mlp_output_dim)

            # aggregate formula logits by node, then normalize, this will give us probilities of each node
            # log p(n) = logsumexp_f log p(f,n)
            frag_node_logits = th.logsumexp(frag_joint_logprobs, dim=1)
            frag_node_logprobs = scatter_logsoftmax(frag_node_logits, frag_node_batch_idxs)

            # calculate conditional probability
            # log p(f|n) = log p(f,n) - log p(n)
            frag_node_formula_logprobs = frag_joint_logprobs - frag_node_logprobs.unsqueeze(-1)
            frag_node_formula_logprobs = th.log_softmax(frag_node_formula_logprobs, dim=1)

        elif self.mlp_output_format == "node_formula":
            # predicts log p(f|n) (formula given node) and log p(n) (node probability) separately.
            # log p(f|n) formula given node
            if self.candidate_ion_head:
                frag_node_formula_logits = self._candidate_ion_logits(
                    frag_embed=frag_embed,
                    frag_x=frag_x,
                    frag_node_feat_idxs=frag_node_feat_idxs,
                    frag_node_batch_idxs=frag_node_batch_idxs,
                    frag_joint_mask=frag_joint_mask,
                    batch_size=batch_size,
                )
            else:
                frag_node_formula_logits = self.formula_module(frag_embed)
            frag_node_formula_logits = self._apply_ion_channel_factorized(
                frag_node_formula_logits, frag_embed, spec_prec_type, frag_node_batch_idxs
            )
            # apply mode-conditioned affine scaling if enabled
            if self.frag_mode_scaler is not None:
                assert spec_frag_mode is not None, "spec_frag_mode required for frag_mode_scaler"
                frag_node_formula_logits = self.frag_mode_scaler(
                    frag_node_formula_logits, spec_frag_mode, frag_node_batch_idxs
                )
            # apply adduct-type FiLM scaling if enabled
            if self.adduct_scaler is not None:
                assert spec_prec_type is not None, "spec_prec_type required for adduct_scaler"
                frag_node_formula_logits = self.adduct_scaler(
                    frag_node_formula_logits, spec_prec_type, frag_node_batch_idxs
                )
            # apply combined stratum FiLM scaling if enabled
            if self.stratum_scaler is not None:
                assert spec_inst_type is not None, "spec_inst_type required for stratum_scaler"
                assert spec_frag_mode is not None, "spec_frag_mode required for stratum_scaler"
                assert spec_prec_type is not None, "spec_prec_type required for stratum_scaler"
                frag_node_formula_logits = self.stratum_scaler(
                    frag_node_formula_logits,
                    spec_inst_type,
                    spec_frag_mode,
                    spec_prec_type,
                    frag_node_batch_idxs,
                )
            assert frag_node_formula_logits.shape[1] == self.mlp_output_dim, (
                frag_node_formula_logits.shape[1],
                self.mlp_output_dim,
            )
            frag_node_formula_logprobs = th.log_softmax(frag_node_formula_logits, dim=1)

            # log p(n) node probability
            frag_node_logits = self.node_module(frag_embed).squeeze(1)
            frag_node_logprobs = scatter_logsoftmax(frag_node_logits, frag_node_batch_idxs)

            # joint logits = formula_logits + node_logits (broadcast); exposed for post-branch
            # scatter_logsoftmax over real positions (mirrors formula path's frag_joint_logits)
            frag_joint_logits = (
                frag_node_formula_logits + frag_node_logits.unsqueeze(-1)
            ).flatten()

            # log p(f,n) = log p(f|n) + log p(n)
            frag_joint_logprobs = frag_node_formula_logprobs + frag_node_logprobs.unsqueeze(-1)
            frag_joint_logprobs = frag_joint_logprobs.flatten()

            # compute total NULL probability (before renormalization)
            frag_null_formula_logprob = scatter_logsumexp(
                (1.0 - frag_joint_mask) * frag_joint_logprobs
                + frag_joint_mask * LOG_ZERO(frag_joint_logprobs.dtype),
                frag_joint_batch_idxs,
            )
            frag_null_formula_logprob = th.clamp(frag_null_formula_logprob, max=0.0)

            if self.mask_null_formula:
                # compute non-NULL renormalized intensity
                frag_joint_logprobs = scatter_masked_softmax(
                    frag_joint_logprobs,
                    frag_joint_mask,
                    frag_joint_batch_idxs,
                    log=True,
                )
        else:
            raise ValueError(f"Unknown mlp_output_format: {self.mlp_output_format}")

        # aggregate by formula
        # log p(f) = logsumexp_n log p(f,n)
        frag_formula_mask = th.ones_like(frag_formula_batch_idxs, dtype=float_dtype)
        frag_formula_mask[frag_formula_cumsizes[:-1]] = 0.0
        # aggregate by formula — use explicit dim_size so dedup'd batches where some high-index
        # formula slots are unreferenced still produce a tensor of size batch_frag_num_formulae,
        # matching frag_formula_mask / frag_formula_batch_idxs.
        frag_formula_logprobs = scatter_logsumexp(
            frag_joint_logprobs.flatten(),
            frag_joint_formula_idxs,
            dim_size=int(batch_frag_num_formulae),
        )
        # softmax over formulae
        if self.mask_null_formula:
            frag_formula_logprobs = scatter_masked_softmax(
                frag_formula_logprobs,
                frag_formula_mask,
                frag_formula_batch_idxs,
                log=True,
            )
        else:
            frag_formula_logprobs = scatter_masked_softmax(
                frag_formula_logprobs,
                th.ones_like(frag_formula_logprobs),
                frag_formula_batch_idxs,
                log=True,
            )

        # OOS Stuff
        if self.predict_oos:
            # get OOS logits and logprobs
            oos_logits = self.oos_module(th.cat([mol_embed_gnn_pool, frag_embed_gnn_pool], dim=1))
            oos_logits = oos_logits.flatten()
            oos_logprobs = F.logsigmoid(oos_logits)
            not_oos_logprobs = F.logsigmoid(-oos_logits)
        else:
            oos_logprobs = th.full(
                [batch_size],
                LOG_ZERO(frag_formula_logprobs.dtype),
                device=device,
                dtype=frag_formula_logprobs.dtype,
            )
            not_oos_logprobs = th.zeros(
                [batch_size], device=device, dtype=frag_formula_logprobs.dtype
            )

        # adjust frag_formula_logprobs
        frag_formula_oos_logprobs = frag_formula_logprobs + th.repeat_interleave(
            not_oos_logprobs, frag_formula_sizes, dim=0
        )

        # convert to spectrum
        frag_formula_offsets = th.repeat_interleave(
            frag_formula_cumsizes[:-1],
            frag_formula_sizes - 1,  # -1 is for NULL formulae
        )
        spec_mzs = frag_formula_peak_mzs
        spec_logprobs = frag_formula_oos_logprobs[
            frag_formula_peak_idxs + frag_formula_offsets
        ] + th.log(frag_formula_peak_probs)

        # get batch idxs
        spec_batch_idxs = th.repeat_interleave(
            th.arange(frag_formula_peak_sizes.shape[0], device=device),
            frag_formula_peak_sizes,
        )

        if not self.skip_edge_loss:
            frag_edge_batch_idxs = frag_batch[frag_edge_index[0]]
            frag_edge_logits = (
                frag_node_logprobs[frag_edge_index[0]] + frag_node_logprobs[frag_edge_index[1]]
            )
            frag_edge_logprobs = scatter_logsoftmax(frag_edge_logits, frag_edge_batch_idxs)
            # print(scatter_logsumexp(frag_edge_logprobs,frag_edge_batch_idxs))
            frag_node_h_counts = get_node_feats(frag_x, frag_node_feat_idxs, "h_counts")
            frag_edge_h_ranges = get_edge_feats(frag_edge_attr, frag_edge_feat_idxs, "h_range")
            # Memory Opt #2: Use einsum to compute h_diffs without Cartesian product expansion
            src_h_counts = frag_node_h_counts[frag_edge_index[0]]  # (num_edges,)
            dst_h_counts = frag_node_h_counts[frag_edge_index[1]]  # (num_edges,)
            frag_edge_h_diffs = src_h_counts.unsqueeze(1) - dst_h_counts.unsqueeze(1)
            frag_edge_h_diffs = frag_edge_h_diffs.reshape(frag_edge_h_diffs.shape[0], -1)
            frag_edge_h_range_masks = th.logical_or(
                frag_edge_h_diffs < frag_edge_h_ranges[:, 0].unsqueeze(-1),
                frag_edge_h_diffs > frag_edge_h_ranges[:, 1].unsqueeze(-1),
            )
            src_formula_logprobs = frag_node_formula_logprobs[
                frag_edge_index[0]
            ]  # (num_edges, mlp_output_dim)
            dst_formula_logprobs = frag_node_formula_logprobs[
                frag_edge_index[1]
            ]  # (num_edges, mlp_output_dim)
            # When CMF is active, only CRF columns represent bond-cleavage H-transfers;
            # CMF columns encode charge-migration which has no edge-H-diff semantics.
            if include_cmf:
                _crf_w = 2 * self.num_hs + 1
                src_formula_logprobs = src_formula_logprobs[:, :_crf_w]
                dst_formula_logprobs = dst_formula_logprobs[:, :_crf_w]
            # Outer product of per-node formula log-probs across each edge: (E, H*H)
            frag_edge_h_logprobs = th.einsum(
                "ni,nj->nij", src_formula_logprobs, dst_formula_logprobs
            ).reshape(len(src_formula_logprobs), -1)

        else:
            frag_edge_logprobs = None
            frag_edge_h_diffs = None
            frag_edge_h_range_masks = None
            frag_edge_h_logprobs = None
            frag_edge_batch_idxs = None

        # Explainability marginals (P(n), P(f), P(f|n), P(n|f), P(h) + nb_iso below) are terminal:
        # nothing on the loss-critical spectrum path (spec_logprobs, null, oos) depends on them.
        # Skip them on TRAINING forwards when opted in (no active consumer) to cut the forward peak;
        # eval/inference (self.training False) and any explainability request always compute them.
        _compute_expl = (
            return_explainability_tensors
            or self.debug_validate_outputs
            or not (self.training and self.skip_explainability_when_training)
        )
        if _compute_expl:
            # select (will remove all NULL formula idxs, potentially some node idxs too if they contain only NULLs)
            _real_mask = frag_joint_mask.bool()
            # Compute real positions once; derive node/h-slot indices via integer div/mod to avoid
            # materializing three separate (N×H) long tensors (node_idxs, h_counts, h_offsets).
            real_pos = th.nonzero(_real_mask, as_tuple=False).squeeze(1)  # (J,)
            H = self.mlp_output_dim
            frag_real_joint_logits = frag_joint_logits[_real_mask]
            frag_real_joint_node_idxs = real_pos // H
            _real_h_pos = real_pos % H
            frag_real_joint_h_counts = self._h_counts_buf[_real_h_pos]
            frag_real_joint_h_offsets = th.zeros_like(frag_real_joint_h_counts)
            frag_real_joint_formula_idxs = frag_joint_formula_idxs[_real_mask]
            frag_real_joint_batch_idxs = frag_node_batch_idxs[frag_real_joint_node_idxs]
            # P(f,n)
            frag_real_joint_logprobs = scatter_logsoftmax(
                frag_real_joint_logits, frag_real_joint_batch_idxs
            )
            # P(n) - sum, renormalize, but keep all-NULL nodes (as zeros)
            _need_expl_idxs = return_explainability_tensors or self.debug_validate_outputs
            frag_real_node_node_idxs = (
                th.arange(batch_frag_num_nodes, device=device) if _need_expl_idxs else None
            )
            frag_real_node_logprobs = scatter_logsumexp(
                frag_real_joint_logprobs,
                frag_real_joint_node_idxs,
                dim_size=batch_frag_num_nodes,
            )
            frag_real_node_logprobs = scatter_logsoftmax(
                frag_real_node_logprobs, frag_node_batch_idxs
            )
            # P(f) - sum, renormalize, but keep NULL formulae (as zeros)
            frag_real_formula_logprobs = scatter_logsumexp(
                frag_real_joint_logprobs,
                frag_real_joint_formula_idxs,
                dim_size=batch_frag_num_formulae,
            )
            frag_real_formula_logprobs = scatter_logsoftmax(
                frag_real_formula_logprobs,
                frag_formula_batch_idxs,
            )
            frag_real_formula_formula_idxs = (
                th.arange(batch_frag_num_formulae, device=device) if _need_expl_idxs else None
            )
            # P(f|n) - remove all NULLs from conditionals
            frag_real_node_formula_logprobs = (
                frag_real_joint_logprobs - frag_real_node_logprobs[frag_real_joint_node_idxs]
            )
            frag_real_node_formula_logprobs = scatter_logsoftmax(
                frag_real_node_formula_logprobs, frag_real_joint_node_idxs
            )
            # P(n|f) - remove all NULLs from conditionals
            frag_real_formula_node_logprobs = (
                frag_real_joint_logprobs - frag_real_formula_logprobs[frag_real_joint_formula_idxs]
            )
            frag_real_formula_node_logprobs = scatter_logsoftmax(
                frag_real_formula_node_logprobs, frag_real_joint_formula_idxs
            )
            # hydrogens
            frag_real_joint_h_idxs = (
                frag_real_joint_h_counts + self.num_hs + frag_real_joint_h_offsets
            ) + frag_real_joint_batch_idxs * self.mlp_output_dim

            frag_real_h_logprobs = scatter_logsumexp(
                frag_real_joint_logprobs,
                frag_real_joint_h_idxs,
                dim_size=batch_size * self.mlp_output_dim,
            )

            frag_real_h_logprobs = th.clamp(frag_real_h_logprobs, max=0.0)
            frag_real_h_counts = self._h_counts_range_buf.repeat(batch_size)
            frag_real_h_batch_idxs = th.repeat_interleave(
                th.arange(batch_size, device=device), self.mlp_output_dim
            )
        else:
            # training fast path: marginals unused (no active consumer) -> emit None
            frag_real_joint_node_idxs = None
            frag_real_joint_formula_idxs = None
            frag_real_joint_batch_idxs = None
            frag_real_joint_logprobs = None
            frag_real_node_node_idxs = None
            frag_real_node_logprobs = None
            frag_real_formula_logprobs = None
            frag_real_formula_formula_idxs = None
            frag_real_node_formula_logprobs = None
            frag_real_formula_node_logprobs = None
            frag_real_joint_h_counts = None
            frag_real_joint_h_idxs = None
            frag_real_h_logprobs = None
            frag_real_h_counts = None
            frag_real_h_batch_idxs = None

        # calculate isomorphic distributions (explainability-only; skipped with the frag_real_*
        # block on the training fast path, where _compute_expl is False)
        if self.nb_iso and _compute_expl:
            # P(n')
            frag_nb_node_logprobs = scatter_logsumexp(
                frag_real_node_logprobs, frag_nb_idxs, dim_size=batch_frag_nb_num_nodes
            )
            frag_nb_node_batch_idxs = scatter_reduce(
                frag_node_batch_idxs,
                frag_nb_idxs,
                reduce="amax",
                dim_size=batch_frag_nb_num_nodes,
            )
            frag_nb_node_node_idxs = frag_nb_un_idxs
            # P(n'|f)
            frag_nb_joint_idxs = frag_nb_idxs[frag_real_joint_node_idxs]
            frag_nb_joint_both_idxs = th.stack(
                [frag_nb_joint_idxs, frag_real_joint_formula_idxs], dim=1
            )
            frag_nb_joint_both_un_idxs, frag_nb_joint_both_inv_idxs = th.unique(
                frag_nb_joint_both_idxs, return_inverse=True, dim=0
            )
            frag_nb_joint_node_idxs = frag_nb_joint_both_un_idxs[:, 0]
            frag_nb_joint_formula_idxs = frag_nb_joint_both_un_idxs[:, 1]
            frag_nb_joint_batch_idxs = frag_nb_node_batch_idxs[frag_nb_joint_node_idxs]
            frag_nb_formula_node_logprobs = scatter_logsumexp(
                frag_real_formula_node_logprobs,
                frag_nb_joint_both_inv_idxs,
                dim_size=frag_nb_joint_both_un_idxs.shape[0],
            )
            frag_nb_formula_node_logprobs = th.clamp(frag_nb_formula_node_logprobs, max=0.0)
            # P(n|n')
            frag_nb_node_node_logprobs = scatter_logsoftmax(frag_real_node_logprobs, frag_nb_idxs)
            frag_nb_node_node_node_idxs = frag_nb_idxs  # frag_nb_node_node_idxs[frag_nb_inv_idxs]
            frag_nb_node_node_batch_idxs = frag_nb_node_batch_idxs[frag_nb_inv_idxs]
            assert th.all(frag_nb_node_node_logprobs <= 0.0)
            # P(f|n')
            frag_nb_node_formula_logprobs = (
                frag_real_node_formula_logprobs
                + frag_nb_node_node_logprobs[frag_real_joint_node_idxs]
            )
            frag_nb_node_formula_logprobs = scatter_logsumexp(
                frag_nb_node_formula_logprobs,
                frag_nb_joint_both_inv_idxs,
                dim_size=frag_nb_joint_both_un_idxs.shape[0],
            )
            frag_nb_node_formula_logprobs = th.clamp(frag_nb_node_formula_logprobs, max=0.0)
            # P(f,n')
            frag_nb_joint_logprobs = scatter_logsumexp(
                frag_real_joint_logprobs,
                frag_nb_joint_both_inv_idxs,
                dim_size=frag_nb_joint_both_un_idxs.shape[0],
            )
        else:
            frag_nb_node_logprobs = None
            frag_nb_node_formula_logprobs = None
            frag_nb_formula_node_logprobs = None
            frag_nb_node_node_logprobs = None
            frag_nb_node_node_idxs = None
            frag_nb_node_batch_idxs = None
            frag_nb_joint_node_idxs = None
            frag_nb_joint_formula_idxs = None
            frag_nb_joint_batch_idxs = None
            frag_nb_joint_logprobs = None
            frag_nb_node_node_node_idxs = None
            frag_nb_node_node_batch_idxs = None

        if self.debug_validate_outputs:
            assert (
                th.unique(frag_real_node_node_idxs).shape[0]
                == frag_real_node_node_idxs.max() + 1
                == batch_frag_num_nodes
            )
            assert (
                th.unique(frag_real_formula_formula_idxs).shape[0]
                == frag_real_formula_formula_idxs.max() + 1
                == batch_frag_num_formulae
            )
            if self.nb_iso:
                assert (
                    th.unique(frag_nb_node_node_idxs).shape[0]
                    == frag_nb_node_node_idxs.max() + 1
                    == batch_frag_nb_num_nodes
                )

        if self.bin_output:
            # import pdb; pdb.set_trace()
            spec_bin_mzs, spec_bin_logprobs, spec_bin_batch_idxs = batched_bin_func(
                mzs=spec_mzs,
                ints=spec_logprobs,
                batch_idxs=spec_batch_idxs,
                mz_max=self.mz_max,
                mz_bin_res=self.mz_bin_res,
                agg="lse",
                sparse=True,
                return_mzs=True,
            )
            spec_mzs = spec_bin_mzs
            spec_logprobs = spec_bin_logprobs
            spec_batch_idxs = spec_bin_batch_idxs

        if self.debug_validate_outputs:
            term1 = scatter_logsumexp(spec_logprobs, spec_batch_idxs).exp()
            term2 = oos_logprobs.exp()
            term3 = (not_oos_logprobs + frag_null_formula_logprob).exp()
            sum_probs = term1 + term2 + term3
            assert th.all(th.isclose(sum_probs, th.ones_like(sum_probs), rtol=0.0, atol=5e-2)), (
                term1,
                term2,
                term3,
            )
        # size definitions:
        # B: batch size
        # P: total number of predicted spectrum peaks across the batch
        # F: total number of predicted formulas across the batch
        # N: total number of predicted nodes (fragments) across the batch
        # J: total number of predicted joint distribution entries across the batch
        # E: total number of edges across the batch
        # H: number of hydrogen counts predicted (num_hs*2+1)
        # N': total number of predicted neighborhood (isomorphic) nodes across the batch
        # J': total number of predicted joint distribution entries for isomorphic nodes across the batch
        out_d = {
            "pred_mzs": spec_mzs,  # Predicted m/z values for the spectrum peaks [P]
            "pred_logprobs": spec_logprobs,  # Predicted log probabilities (intensities) for the spectrum peaks [P]
            "pred_batch_idxs": spec_batch_idxs,  # Batch indices mapping each peak to a spectrum in the batch [P]
            "pred_formula_logprobs": frag_real_formula_logprobs,  # Log probabilities of formulas P(f) [F]
            "pred_formula_batch_idxs": frag_formula_batch_idxs,  # Batch indices for the formulas [F]
            "pred_node_logprobs": frag_real_node_logprobs,  # Log probabilities of nodes (fragments) P(n) [N]
            "pred_node_batch_idxs": frag_node_batch_idxs,  # Batch indices for the nodes [N]
            "pred_node_formula_logprobs": frag_real_node_formula_logprobs,  # Conditional log probabilities P(f|n) [J]
            "pred_formula_node_logprobs": frag_real_formula_node_logprobs,  # Conditional log probabilities P(n|f) [J]
            "pred_joint_logprobs": frag_real_joint_logprobs,  # Joint log probabilities P(f,n) [J]
            "pred_joint_node_idxs": frag_real_joint_node_idxs,  # Node indices for the joint distribution entries [J]
            "pred_joint_formula_idxs": frag_real_joint_formula_idxs,  # Formula indices for the joint distribution entries [J]
            "pred_joint_batch_idxs": frag_real_joint_batch_idxs,  # Batch indices for the joint distribution entries [J]
            "pred_null_formula_logprob": frag_null_formula_logprob,  # Log probability of the NULL formula (unexplained intensity) [B]
            "pred_edge_logprobs": frag_edge_logprobs,  # Log probabilities for edges (if edge loss is used) [E]
            "pred_edge_h_diffs": frag_edge_h_diffs,  # Hydrogen count differences across edges [E, H*H]
            "pred_edge_h_range_masks": frag_edge_h_range_masks,  # Masks for valid hydrogen ranges on edges [E, H*H]
            "pred_edge_h_logprobs": frag_edge_h_logprobs,  # Log probabilities for hydrogen transfers on edges [E, H*H]
            "pred_edge_batch_idxs": frag_edge_batch_idxs,  # Batch indices for edges [E]
            "pred_oos_logprobs": oos_logprobs,  # Log probability of Out-Of-Scope (OOS) intensity [B]
            "pred_h_counts": frag_real_h_counts,  # Hydrogen counts for the hydrogen distribution [B*H]
            "pred_h_batch_idxs": frag_real_h_batch_idxs,  # Batch indices for the hydrogen distribution [B*H]
            "pred_h_logprobs": frag_real_h_logprobs,  # Log probabilities of hydrogen counts [B*H]
            "pred_nb_node_logprobs": frag_nb_node_logprobs,  # (Isomorphism) Log probabilities of neighborhood nodes P(n') [N']
            "pred_nb_node_formula_logprobs": frag_nb_node_formula_logprobs,  # (Isomorphism) Conditional log probabilities P(f|n') [J']
            "pred_nb_formula_node_logprobs": frag_nb_formula_node_logprobs,  # (Isomorphism) Conditional log probabilities P(n'|f) [J']
            "pred_nb_node_node_logprobs": frag_nb_node_node_logprobs,  # (Isomorphism) Conditional log probabilities P(n|n') [N]
            "pred_nb_node_node_idxs": frag_nb_node_node_idxs,  # (Isomorphism) Indices for node-node mapping [N]
            "pred_nb_node_batch_idxs": frag_nb_node_batch_idxs,  # (Isomorphism) Batch indices for neighborhood nodes [N']
            "pred_nb_joint_logprobs": frag_nb_joint_logprobs,  # (Isomorphism) Joint log probabilities P(f,n') [J']
            "pred_nb_joint_node_idxs": frag_nb_joint_node_idxs,  # (Isomorphism) Node indices for joint distribution [J']
            "pred_nb_joint_formula_idxs": frag_nb_joint_formula_idxs,  # (Isomorphism) Formula indices for joint distribution [J']
            "pred_nb_joint_batch_idxs": frag_nb_joint_batch_idxs,  # (Isomorphism) Batch indices for joint distribution [J']
            "pred_nb_node_node_node_idxs": frag_nb_node_node_node_idxs,  # (Isomorphism) Node indices for node-node mapping [N]
            "pred_nb_node_node_batch_idxs": frag_nb_node_node_batch_idxs,  # (Isomorphism) Batch indices for node-node mapping [N]
        }
        if return_explainability_tensors:
            out_d["pred_formula_formula_idxs"] = frag_real_formula_formula_idxs  # [F]
            out_d["pred_node_node_idxs"] = frag_real_node_node_idxs  # [N]
            out_d["pred_joint_h_counts"] = frag_real_joint_h_counts  # [J]
            out_d["pred_joint_h_idxs"] = frag_real_joint_h_idxs  # [J]

        if self.output_formula_str:
            assert frag_formula_str is not None, (
                "frag_formula_strs must be provided if output_formula_str=True"
            )
            assert spec_prec_type_str is not None, (
                "spec_prec_type_strs must be provided if output_formula_str=True"
            )
            assert frag_formula_str.shape[0] == frag_real_formula_logprobs.shape[0]
            prec_type_str = spec_prec_type_str
            prec_type_delta_comp = np.array(
                [PREC_TYPE_TO_FORMULA_DIFF[prec_type_str[i]] for i in range(len(prec_type_str))]
            )
            prec_type_delta_comp = prec_type_delta_comp[frag_formula_batch_idxs.cpu().numpy()]
            pred_formula_str = [
                combine_formulae(frag_formula_str[i], prec_type_delta_comp[i])
                if frag_formula_str[i] != ""
                else ""
                for i in range(len(frag_formula_str))
            ]
            pred_formula_str = np.array(pred_formula_str)
            out_d["pred_formula_str"] = pred_formula_str

        if self.debug_validate_outputs:
            for k, v in out_d.items():
                if "logprob" in k and v is not None and v.max() > 0.0:
                    logger.warning(f"{k} has value {v.max()} > 0")
                if "batch_idx" in k and v is not None:
                    if v.numel() == 0:
                        raise ValueError(f"Empty batch index: {k}")
                    elif th.unique(v).shape[0] != batch_size:
                        raise ValueError(
                            f"Missing items in {k} of batch: {th.unique(v).shape[0]} expeacting {batch_size}"
                        )
        return out_d
