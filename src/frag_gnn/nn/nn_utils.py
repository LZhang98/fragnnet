import copy
import math
import numpy as np
import scipy.sparse as sparse
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as th_s

from dgl.backend import pytorch as dgl_F
import frag_gnn.nn.dgl_modules as dgl_mods
from frag_gnn.nn.form_embedder import get_embedder


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_lr_scheduler(
    optimizer, lr_decay_rate: float, decay_steps: int = 5000, warmup: int = 1000
):
    """build_lr_scheduler.

    Args:
        optimizer:
        lr_decay_rate (float): lr_decay_rate
        decay_steps (int): decay_steps
        warmup_steps (int): warmup_steps
    """

    def lr_lambda(step):
        if step >= warmup:
            # Adjust
            step = step - warmup
            rate = lr_decay_rate ** (step // decay_steps)
        else:
            rate = 1 - math.exp(-step / warmup)
        return rate

    scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler

class JointPredictor(nn.Module):

    # data: mol dgl, dag dgl, peaks arrays
    # TODO: add reverse edges?
    # TODO: allow for more sophisticated formula embeddings

    def __init__(
        self,
        num_depth,
        num_hs,
        num_elements,
        mol_kwargs,
        dag_kwargs,
        mlp_kwargs,
        int_embedder
    ):

        super().__init__()
        self.num_depth = num_depth
        self.num_hs = num_hs
        self.num_elements = num_elements
        self.mol_embedder = GNN(**mol_kwargs)
        self.formula_embedder = get_embedder(int_embedder)
        self.depth_embedder = get_embedder(int_embedder)
        # molecule, formula, depth
        self.dag_node_feats = self.mol_embedder.hidden_size \
            + self.num_elements*self.formula_embedder.num_dim \
            + self.num_depth*self.depth_embedder.num_dim
        # molecule, formula
        self.dag_edge_feats = self.mol_embedder.hidden_size \
            + self.num_elements*self.formula_embedder.num_dim
        dag_kwargs = {
            "gnn_node_feats": self.dag_node_feats,
            "gnn_edge_feats": self.dag_edge_feats,
            **dag_kwargs
        }
        self.dag_embedder = GNN(**dag_kwargs)
        mlp_kwargs = {
            "input_size": self.dag_embedder.hidden_size,
            "output_size": 2*self.num_hs+1,
            **mlp_kwargs
        }
        self.output_module = MLPBlocks(**mlp_kwargs)

    def forward(self, batched_dag):

        # N is batch dimension (assume 1 for now)
        # D is number of nodes in the DAG
        # A is number of atoms
        # M is integer embedding dimension
        # H is hidden dimension of atom embeddings
        # T is the number of Hs transferred
        # F is the maximum number of formulae per DAG
        # E is the number of elements (heavy and H)
  
        mol_dgl, dag_dgl = batched_dag.get_batched_dgls()
        mol_nodes, dag_nodes = batched_dag.get_batched_nodes()
        dag_formula_peak_mzs = batched_dag.batched_formula_peak_mzs
        dag_formula_peak_probs = batched_dag.batched_formula_peak_probs
        dag_formula_peak_sizes = batched_dag.batched_formula_peak_sizes
        dag_formula_cumsizes = batched_dag.batched_formula_cumsizes
        dag_formula_peak_idxs = batched_dag.batched_formula_peak_idxs

        # get per-atom embeddings [N,A,H]
        mol_embed = self.mol_embedder(mol_dgl)
        
        # process dag
        with dag_dgl.local_scope():
            
            dag_ndata, dag_edata = [], []
            dag_num_nodes = dag_dgl.number_of_nodes()
            dag_num_edges = dag_dgl.number_of_edges()
            # get atom embeddings for each node
            dag_node_mask_idxs = th.nonzero(dag_dgl.ndata.pop('mask')).long()
            dag_node_offsets = mol_nodes[th.bucketize(dag_node_mask_idxs[:,0],dag_nodes,right=True)-1]
            dag_node_mask_idxs[:,1] = dag_node_mask_idxs[:,1] + dag_node_offsets
            dag_node_mask_embed = th_s.scatter_add(
                mol_embed[dag_node_mask_idxs[:,1]],
                dag_node_mask_idxs[:,0],
                dim=0,
                dim_size=dag_num_nodes
            )
            # print(dag_node_mask_embed.shape)
            dag_ndata.append(dag_node_mask_embed)
            # do the same for edges
            dag_edge_mask_idxs = th.nonzero(dag_dgl.edata.pop('mask')).long()
            dag_edge_node_idxs = dag_dgl.edges()[0][dag_edge_mask_idxs[:,0]]
            dag_edge_offsets = mol_nodes[th.bucketize(dag_edge_node_idxs,dag_nodes,right=True)-1]
            dag_edge_mask_idxs[:,1] = dag_edge_mask_idxs[:,1] + dag_edge_offsets
            dag_edge_mask_embed = th_s.scatter_add(
                mol_embed[dag_edge_mask_idxs[:,1]],
                dag_edge_mask_idxs[:,0],
                dim=0,
                dim_size=dag_num_edges
            )
            # print(dag_edge_mask_embed.shape)
            dag_edata.append(dag_edge_mask_embed)
            # embed node formulae
            dag_node_formula = self.formula_embedder(dag_dgl.ndata.pop('base_formula')).reshape(dag_num_nodes,-1)
            dag_ndata.append(dag_node_formula)
            # do the same for edges
            dag_edge_formula = self.formula_embedder(dag_dgl.edata.pop('base_formula')).reshape(dag_num_edges,-1)
            dag_edata.append(dag_edge_formula)
            # embed node depth
            dag_depth = self.depth_embedder(dag_dgl.ndata.pop('depth')).reshape(dag_num_nodes,-1)
            dag_ndata.append(dag_depth)
            # concatenate everything [N,H+(E+1)*M]
            dag_dgl.ndata['h'] = th.cat(dag_ndata,dim=-1)
            # concatenate everything [N,H+E*M]
            dag_dgl.edata['e'] = th.cat(dag_edata,dim=-1)
            # apply GNN
            dag_embed = self.dag_embedder(dag_dgl)

            # predict probabilities [N,2*T+1]
            dag_formula_probs = self.output_module(dag_embed)
            # aggregate by formula [N,F]
            dag_formula_batch_idxs = th.bucketize(dag_dgl.nodes(),dag_nodes,right=True)-1
            dag_formula_offsets = dag_formula_cumsizes[dag_formula_batch_idxs]
            dag_formula_idxs = (dag_dgl.ndata.pop('h_formulae_idx')+dag_formula_offsets.unsqueeze(-1)).flatten()
            dag_formula_probs = dag_formula_probs.flatten()
            dag_formula_probs = th_s.scatter_add(
                dag_formula_probs, 
                dag_formula_idxs,
                dim=0,
                dim_size=dag_formula_cumsizes[-1]
            )
            # zero out NULL formula
            dag_formula_mask = th.ones_like(dag_formula_probs)
            dag_formula_mask[dag_formula_cumsizes[:-1]] = 0.
            dag_formula_batch_idxs = th.bucketize(
                th.arange(dag_formula_cumsizes[-1],device=dag_formula_cumsizes.device),
                dag_formula_cumsizes,
                right=True
            )-1
            # softmax with masked logits
            dag_formula_probs = masked_subset_softmax(
                dag_formula_probs,
                dag_formula_mask,
                dag_formula_batch_idxs,
                dim=0,
                log=True
            )
            # convert to spectrum
            spec_mzs = dag_formula_peak_mzs
            spec_logprobs = dag_formula_probs[dag_formula_peak_idxs] + th.log(dag_formula_peak_probs)
        # check that the total intensity sums up to 1 (approximately)
        spec_batch_idxs = th.repeat_interleave(
            th.arange(dag_formula_peak_sizes.shape[0],device=dag_formula_peak_sizes.device),
            dag_formula_peak_sizes
        )
        
        return spec_mzs, spec_logprobs, spec_batch_idxs

class GNN(nn.Module):
    """MoleculeGNN Module"""

    def __init__(
        self,
        hidden_size: int,
        num_step_message_passing: int = 4,
        gnn_node_feats: int = 74,
        gnn_edge_feats: int = 4,  # 12,
        mpnn_type: str = "GINE",
        node_feat_symbol = "h",
        dropout: float = 0,
        **kwargs
    ):
        """__init__.
        Args:
            hidden_size (int): Hidden size
            num_mol_layers (int): Number of layers to encode for the molecule
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.gnn_edge_feats = gnn_edge_feats
        self.gnn_node_feats = gnn_node_feats
        self.node_feat_symbol = node_feat_symbol
        self.dropout = dropout

        self.mpnn_type = mpnn_type
        self.hidden_size = hidden_size
        self.num_step_message_passing = num_step_message_passing
        self.input_project = nn.Linear(self.gnn_node_feats, self.hidden_size)

        if self.mpnn_type == "GGNN":
            self.gnn = GGNN(
                hidden_size=self.hidden_size,
                edge_feats=self.gnn_edge_feats,
                node_feats=self.gnn_node_feats,
                num_step_message_passing=num_step_message_passing,
            )
        elif self.mpnn_type == "PNA":
            self.gnn = PNA(
                hidden_size=self.hidden_size,
                edge_feats=self.gnn_edge_feats,
                node_feats=self.gnn_node_feats,
                num_step_message_passing=num_step_message_passing,
                dropout=self.dropout,
            )
        elif self.mpnn_type == "GINE":
            self.gnn = GINE(
                hidden_size=self.hidden_size,
                edge_feats=self.gnn_edge_feats,
                node_feats=self.gnn_node_feats,
                num_step_message_passing=num_step_message_passing,
                dropout=self.dropout,
            )
        else:
            raise ValueError()

    def forward(self, g):
        """encode batch of molecule graph"""
        with g.local_scope():
            # Set initial hidden
            ndata = g.ndata[self.node_feat_symbol]
            edata = g.edata["e"]
            h_init = self.input_project(ndata)
            g.ndata.update({"_h": h_init})
            g.edata.update({"_e": edata})

            if self.mpnn_type == "GGNN":
                # Get graph output
                output = self.gnn(g, "_h", "_e")
            elif self.mpnn_type == "PNA":
                # Get graph output
                output = self.gnn(g, "_h", "_e")
            elif self.mpnn_type == "GINE":
                # Get graph output
                output = self.gnn(g, "_h", "_e")
            else:
                raise NotImplementedError()

        return output


class GINE(nn.Module):
    def __init__(
        self,
        hidden_size=64,
        edge_feats=4,
        num_step_message_passing=4,
        dropout=0,
        **kwargs
    ):
        """GINE.

        Args:
            input_size (int): Size of edge features into the graph
            hidden_size (int): Hidden size
            edge_feats (int): Number of edge feats. Must be onehot!
            node_feats (int): Num of node feats (default 74)
            num_step_message_passing (int): Number of message passing steps
            dropout
        """
        super().__init__()

        self.edge_transform = nn.Linear(edge_feats, hidden_size)

        self.layers = []
        for i in range(num_step_message_passing):
            apply_fn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            temp_layer = dgl_mods.GINEConv(apply_func=apply_fn, init_eps=0)
            self.layers.append(temp_layer)

        self.layers = nn.ModuleList(self.layers)
        self.bnorms = get_clones(nn.BatchNorm1d(hidden_size), num_step_message_passing)
        self.dropouts = get_clones(nn.Dropout(dropout), num_step_message_passing)

    def forward(self, graph, nfeat_name="_h", efeat_name="_e"):
        """forward.

        Args:
            graph (dgl graph): Graph object
            nfeat_name (str): Name of node feat data
            efeat_name (str): Name of e feat

        Return:
            h: Hidden state at each node


        """
        node_feat, edge_feat = graph.ndata[nfeat_name], graph.edata[efeat_name]
        edge_feat = self.edge_transform(edge_feat)

        for dropout, layer, norm in zip(self.dropouts, self.layers, self.bnorms):
            layer_out = layer(graph, node_feat, edge_feat)
            node_feat = F.relu(dropout(norm(layer_out))) + node_feat

        return node_feat


class GGNN(nn.Module):
    def __init__(
        self, hidden_size=64, edge_feats=4, num_step_message_passing=4, **kwargs
    ):
        """GGNN.

        Define a gated graph neural network

        This is very similar to the NNConv models.

        Args:
            input_size (int): Size of edge features into the graph
            hidden_size (int): Hidden size
            edge_feats (int): Number of edge feats. Must be onehot!
            node_feats (int): Num of node feats (default 74)
            num_step_message_passing (int): Number of message passing steps
        """
        super().__init__()
        self.model = dgl_mods.GatedGraphConv(
            in_feats=hidden_size,
            out_feats=hidden_size,
            n_steps=num_step_message_passing,
            n_etypes=edge_feats,
        )

    def forward(self, graph, nfeat_name="_h", efeat_name="_e"):
        """forward.

        Args:
            graph (dgl graph): Graph object
            nfeat_name (str): Name of node feat data
            efeat_name (str): Name of e feat

        Return:
            h: Hidden state at each node

        """
        if "e_ind" in graph.edata:
            etypes = graph.edata["e_ind"]
        else:
            etypes = graph.edata[efeat_name].argmax(1)
        return self.model(graph, graph.ndata[nfeat_name], etypes=etypes)


class PNA(nn.Module):
    def __init__(
        self,
        hidden_size=64,
        edge_feats=4,
        num_step_message_passing=4,
        dropout=0,
        **kwargs
    ):
        """PNA.

        Define a PNA network

        Args:
            input_size (int): Size of edge features into the graph
            hidden_size (int): Hidden size
            edge_feats (int): Number of edge feats. Must be onehot!
            node_feats (int): Num of node feats (default 74)
            num_step_message_passing (int): Number of message passing steps
        """
        super().__init__()
        self.layer = dgl_mods.PNAConv(
            in_size=hidden_size,
            out_size=hidden_size,
            aggregators=["mean", "max", "min", "std", "var", "sum"],
            scalers=["identity", "amplification", "attenuation"],
            delta=2.5,
            dropout=dropout,
        )

        self.layers = get_clones(self.layer, num_step_message_passing)
        self.bnorms = get_clones(nn.BatchNorm1d(hidden_size), num_step_message_passing)

    def forward(self, graph, nfeat_name="_h", efeat_name="_e"):
        """forward.

        Args:
            graph (dgl graph): Graph object
            nfeat_name (str): Name of node feat data
            efeat_name (str): Name of e feat

        Return:
            h: Hidden state at each node

        """
        node_feat, edge_feat = graph.ndata[nfeat_name], graph.edata[efeat_name]
        for layer, norm in zip(self.layers, self.bnorms):
            node_feat = F.relu(norm(layer(graph, node_feat, edge_feat))) + node_feat

        return node_feat


class MLPBlocks(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        num_layers: int,
        output_size: int = None,
        use_residuals: bool = False,
    ):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(input_size, hidden_size)
        middle_layer = nn.Linear(hidden_size, hidden_size)
        self.layers = get_clones(middle_layer, num_layers - 1)

        self.output_layer = None
        self.output_size = output_size
        if self.output_size is not None:
            self.output_layer = nn.Linear(hidden_size, self.output_size)

        self.use_residuals = use_residuals

    def forward(self, x):
        output = x
        output = self.input_layer(x)
        output = self.dropout_layer(output)
        output = self.activation(output)
        old_op = output
        for layer_index, layer in enumerate(self.layers):
            output = layer(output)
            output = self.dropout_layer(output)
            output = self.activation(output)
            if self.use_residuals:
                output = output + old_op
                old_op = output

        if self.output_layer is not None:
            output = self.output_layer(output)
        return output


# DGL Models
# https://docs.dgl.ai/en/0.6.x/_modules/dgl/nn/pytorch/glob.html#SetTransformerDecoder


def random_walk_pe(g, k, eweight_name=None):
    """Random Walk Positional Encoding, as introduced in
    `Graph Neural Networks with Learnable Structural and Positional Representations
    <https://arxiv.org/abs/2110.07875>`__

    This function computes the random walk positional encodings as landing probabilities
    from 1-step to k-step, starting from each node to itself.

    Parameters
    ----------
    g : DGLGraph
        The input graph. Must be homogeneous.
    k : int
        The number of random walk steps. The paper found the best value to be 16 and 20
        for two experiments.
    eweight_name : str, optional
        The name to retrieve the edge weights. Default: None, not using the edge weights.

    Returns
    -------
    Tensor
        The random walk positional encodings of shape :math:`(N, k)`, where :math:`N` is the
        number of nodes in the input graph.

    Example
    -------
    >>> import dgl
    >>> g = dgl.graph(([0,1,1], [1,1,0]))
    >>> dgl.random_walk_pe(g, 2)
    tensor([[0.0000, 0.5000],
            [0.5000, 0.7500]])
    """
    N = g.num_nodes()  # number of nodes
    M = g.num_edges()  # number of edges
    A = g.adj(scipy_fmt="csr")  # adjacency matrix
    if eweight_name is not None:
        # add edge weights if required
        W = sparse.csr_matrix(
            (g.edata[eweight_name].squeeze(), g.find_edges(list(range(M)))),
            shape=(N, N),
        )
        A = A.multiply(W)
    RW = np.array(A / (A.sum(1) + 1e-30))  # 1-step transition probability

    # Iterate for k steps
    PE = [dgl_F.astype(dgl_F.tensor(np.copy(RW.diagonal())), th.float32)]
    RW_power = RW
    for _ in range(k - 1):
        RW_power = RW_power @ RW
        PE.append(dgl_F.astype(dgl_F.tensor(np.copy(RW_power.diagonal())), th.float32))
    PE = dgl_F.stack(PE, dim=-1)
    return PE

def masked_subset_softmax(logits,mask,subset_idxs,mask_value=-1e9,dim=-1,log=True):
    # apply mask
    masked_logits = mask*logits + (1-mask)*mask_value
    # normalize
    masked_logits = scatter_normalize(masked_logits,subset_idxs,dim)
    if not log:
        # exponentiate
        return th.exp(masked_logits)
    else:
        return masked_logits

def scatter_normalize(logits,subset_idxs,dim=-1):
    # calculate normalizing constant
    c = th_s.scatter_logsumexp(
        logits,
        subset_idxs,
        dim=dim
    )
    # apply normalizing constant
    logits = logits - c[subset_idxs]
    return logits