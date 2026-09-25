"""Reproduce the MASSter dropout / normalization / generalization audit probes.

Every probe is self-contained and needs no dataset or checkpoint: each one builds a
randomly initialized, production-shaped module and measures a mechanism. Numbers here
establish direction and mechanism, not calibrated magnitudes for a trained model.

Usage:
    python scripts/debug/probe_masster_regularization.py            # mechanism probes
    python scripts/debug/probe_masster_regularization.py --wandb    # + W&B run census

Probes:
    1. Dropout module inventory for the production encoder.
    2. Dropout placement: relu(dropout(x)) versus dropout(relu(x)).
    3. Residual-branch attenuation under branch-normalization.
    4. Batch-composition leakage and train/eval divergence, by normalization.
    5. Train-versus-eval candidate-set agreement, isolating BatchNorm from dropout.
    6. Whether ``lr_schedule_type: cosine`` reaches the scheduler.
    7. Weight decay applied to normalization gains, biases and embeddings.
    8. Post-LN versus pre-LN gradient profile in the candidate-attention stack.
    9. Normalization inventory outside the GNN, and bond-head residual attenuation.
   10. Context-vector scale versus the merged CE-reading count.
"""

from __future__ import annotations

import argparse
import collections
import sys

import torch as th
import torch.nn as nn

from fragnnet.model import MassterModel, bond_fragment_heads
from fragnnet.model.bond_fragment_heads import BondFocusedBreakHead
from fragnnet.model.masster_model import MassterContextEncoder
from fragnnet.model.nn_blocks import GNN
from fragnnet.utils.feat_utils import get_mol_feats_sizes
from fragnnet.utils.nn_utils import build_lr_scheduler
from fragnnet.utils.pl_utils import LEGACY_MASSTER_MODEL_TYPE, migrate_legacy_masster_name

# The production molecular feature set used by the merged MASSter configs.
NODE_FEATS = [
    "a_onehot", "a_degree", "a_hybrid", "a_formal", "a_radical",
    "a_ring", "a_ring_size", "a_num_hs", "a_mass", "a_chiral",
]
EDGE_FEATS = ["b_degree", "b_ring", "b_conj"]
NORMALIZATIONS = ("batch", "layer", "graph", "none")
# Aspirin and caffeine: drug-sized, so top-K candidate selection has real choices.
PROBE_SMILES = ("CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C")


def _banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _production_encoder(normalization: str, dropout: float, node_dim: int, edge_dim: int) -> GNN:
    """Build the encoder shape used by the merged configs (GINE_GSA 384x2)."""
    th.manual_seed(11)
    return GNN(
        hidden_size=384,
        num_layers=2,
        node_feats_size=node_dim,
        edge_feats_size=edge_dim,
        gnn_type="GINE_GSA",
        dropout=dropout,
        normalization=normalization,
        num_heads=8,
    )


def _block_diagonal_batch(
    sizes: list[int],
    node_dim: int,
    edge_dim: int,
    seed: int,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Build a molecular mini-batch whose bonds never cross molecules.

    Args:
        sizes: Atom count per molecule.
        node_dim: Node feature width.
        edge_dim: Edge feature width.
        seed: Generator seed.

    Returns:
        Node features ``(num_atoms, node_dim)``, batch index ``(num_atoms,)``,
        edge index ``(2, num_edges)``, and edge features ``(num_edges, edge_dim)``.
    """
    generator = th.Generator().manual_seed(seed)
    batch_index: list[int] = []
    sources: list[int] = []
    targets: list[int] = []
    offset = 0
    for molecule_index, num_atoms in enumerate(sizes):
        batch_index += [molecule_index] * num_atoms
        # A spanning tree keeps every molecule connected without crossing molecules.
        for atom in range(1, num_atoms):
            parent = int(th.randint(0, atom, (1,), generator=generator))
            sources += [offset + atom, offset + parent]
            targets += [offset + parent, offset + atom]
        offset += num_atoms
    edge_index = th.stack([th.tensor(sources), th.tensor(targets)])
    return (
        th.randn(offset, node_dim, generator=generator),
        th.tensor(batch_index),
        edge_index,
        th.randn(edge_index.shape[1], edge_dim, generator=generator),
    )


def probe_dropout_inventory(node_dim: int, edge_dim: int) -> None:
    _banner("PROBE 1: dropout and normalization module inventory, production encoder")
    encoder = _production_encoder("batch", 0.1, node_dim, edge_dim)
    dropouts = [(n, m.p) for n, m in encoder.named_modules() if isinstance(m, nn.Dropout)]
    norms = [
        (n, type(m).__name__)
        for n, m in encoder.named_modules()
        if type(m).__name__ in ("BatchNorm1d", "LayerNorm", "GraphNorm")
    ]
    print(f"  GINE_GSA 384x2: {len(dropouts)} Dropout modules, p={sorted({p for _, p in dropouts})}")
    print(f"                  {len(norms)} norm modules, {sorted({t for _, t in norms})}")
    on_input = any(name.startswith("input_project") for name, _ in dropouts)
    print(f"  dropout on the input projection: {on_input}  (no input-feature dropout exists)")


def probe_dropout_placement() -> None:
    _banner("PROBE 2: dropout placement, relu(dropout(x)) versus dropout(relu(x))")
    features = th.randn(4096, 32)
    dropout = nn.Dropout(0.1)
    th.manual_seed(7)
    before = th.relu(dropout(features))
    th.manual_seed(7)
    after = dropout(th.relu(features))
    print(f"  max absolute difference: {(before - after).abs().max().item():.3e}")
    print(f"  equivalent: {th.allclose(before, after)}  (ReLU is positively homogeneous)")


def probe_residual_attenuation(node_dim: int, edge_dim: int) -> None:
    _banner("PROBE 3: can a layer attenuate its own residual branch? (train mode)")
    print("  ||branch|| / ||residual stream||, per layer, versus input magnitude")
    for gnn_type in ("GINE", "GINE_GSA"):
        print(f"\n  {gnn_type}")
        for normalization in NORMALIZATIONS:
            columns = []
            for scale in (0.1, 1.0, 10.0):
                encoder = _production_encoder(normalization, 0.0, node_dim, edge_dim)
                encoder.gnn_type = gnn_type
                encoder = GNN(
                    hidden_size=384, num_layers=2, node_feats_size=node_dim,
                    edge_feats_size=edge_dim, gnn_type=gnn_type, dropout=0.0,
                    normalization=normalization, num_heads=8,
                ).train()
                x, batch, edge_index, edge_attr = _block_diagonal_batch(
                    [30] * 20, node_dim, edge_dim, seed=3
                )
                core = encoder.gnn
                ratios = []
                with th.no_grad():
                    state = encoder.input_project(x * scale)
                    edges = core.edge_transform(edge_attr)
                    for dropout, layer, norm in zip(core.dropouts, core.layers, core.norms):
                        normed = core.norm_fn(layer(state, edge_index, edges), batch, norm)
                        branch = th.relu(dropout(normed))
                        ratios.append((branch.norm() / state.norm()).item())
                        state = branch + state
                columns.append(f"scale={scale:<5g} {[f'{v:.3f}' for v in ratios]}")
            print(f"    norm={normalization:6s} " + "   ".join(columns))
    print("\n  Any normalization on the branch pins ||branch|| independently of the stream:")
    print("  the ratio tracks 1/scale exactly. Only norm=none lets a layer attenuate itself.")


def probe_batch_composition(node_dim: int, edge_dim: int) -> None:
    _banner("PROBE 4: batch-composition leakage and train/eval divergence")
    print("  Molecule 0 embedded (a) among 19 companions, (b) alone. Identical subgraph.")
    for normalization in NORMALIZATIONS:
        encoder = _production_encoder(normalization, 0.0, node_dim, edge_dim)
        x, batch, edge_index, edge_attr = _block_diagonal_batch(
            [30] * 20, node_dim, edge_dim, seed=3
        )
        encoder.train()
        with th.no_grad():
            for _ in range(300):  # converge BatchNorm running statistics
                encoder(x + 0.05 * th.randn_like(x), batch, edge_index, edge_attr)
            keep = batch == 0
            edge_keep = keep[edge_index[0]] & keep[edge_index[1]]
            in_batch = encoder(x, batch, edge_index, edge_attr)[keep]
            alone = encoder(
                x[keep], batch[keep], edge_index[:, edge_keep], edge_attr[edge_keep]
            )
            train_output = encoder(x, batch, edge_index, edge_attr)
        encoder.eval()
        with th.no_grad():
            eval_output = encoder(x, batch, edge_index, edge_attr)
        print(
            f"  norm={normalization:6s}"
            f"  companions move molecule 0: {((in_batch - alone).norm() / alone.norm()).item():.4f}"
            f"   train vs eval, same batch: "
            f"{((train_output - eval_output).norm() / eval_output.norm()).item():.4f}"
        )

    print("\n  Molecule-size sensitivity: molecule 0 held byte-identical, companions grow 30->120.")
    for normalization in NORMALIZATIONS:
        encoder = _production_encoder(normalization, 0.0, node_dim, edge_dim).train()
        x_small, b_small, ei_small, _ = _block_diagonal_batch(
            [30] * 20, node_dim, edge_dim, seed=3
        )
        x_large, b_large, ei_large, _ = _block_diagonal_batch(
            [30] + [120] * 19, node_dim, edge_dim, seed=3
        )
        molecule_0_edges = int((b_small[ei_small[0]] == 0).sum())
        # The generator emits molecule 0 first in both batches, so its subgraph matches.
        assert th.equal(ei_small[:, :molecule_0_edges], ei_large[:, :molecule_0_edges])
        x_large[:30] = x_small[:30]
        ea_small = th.randn(ei_small.shape[1], edge_dim, generator=th.Generator().manual_seed(5))
        ea_large = th.randn(ei_large.shape[1], edge_dim, generator=th.Generator().manual_seed(5))
        ea_large[:molecule_0_edges] = ea_small[:molecule_0_edges]
        with th.no_grad():
            small = encoder(x_small, b_small, ei_small, ea_small)[:30]
            large = encoder(x_large, b_large, ei_large, ea_large)[:30]
        print(
            f"  norm={normalization:6s}  ||delta molecule 0|| / ||molecule 0||: "
            f"{((small - large).norm() / small.norm()).item():.4f}"
        )


def _build_probe_model(dropout: float, normalization: str, node_dim: int, edge_dim: int):
    """Production-shaped MASSter model matching the merged unique-fragment configs."""
    th.manual_seed(903)
    return MassterModel(
        node_feature_dim=node_dim, edge_feature_dim=edge_dim,
        precursor_hidden_dim=384, precursor_num_layers=2, precursor_num_heads=8,
        precursor_gnn_type="GINE_GSA", precursor_normalization=normalization,
        precursor_dropout=dropout, context_dim=128, context_in_gnn=False,
        metadata_embedding_dim=32, head_hidden_dim=512, head_dropout=dropout,
        bond_num_layers=3, intensity_transformer_dim=384,
        intensity_transformer_num_layers=3, intensity_transformer_num_heads=8,
        intensity_transformer_feedforward_dim=1024,
        intensity_transformer_num_inducing_points=128,
        intensity_transformer_attention="induced",
        intensity_transformer_dropout=dropout, intensity_output="probability",
        use_mask_residuals=True, proposal_selection_only=True,
        separate_proposal_encoder=True, precursor_types=("[M+H]+",),
        instrument_types=[], fragmentation_modes=(), int_embedder="abs-sines",
        ce_insert_type="embed", ce_insert_size=6, use_nce=True, use_ace=False,
        nce_max=200.0, nce_mean=60.0, nce_std=40.0,
        ace_max=100.0, ace_mean=20.0, ace_std=15.0,
        max_h_transfer=4, include_cmf=True, proposal_num_mixtures=1,
        use_base_probability_feature=False, return_base_spectrum=True,
        detach_base_ion_h=True, share_proposal_across_ev=False,
        separate_ion_mode_proposal_head=False, ion_mode_proposal_binary_logits=False,
        min_ion_mode_relative_probability=0.0, separate_ion_h_support_head=False,
        ion_h_support_binary_logits=False, gate_independent_intensity_with_support=False,
        min_ion_h_relative_probability=0.0, min_ion_h_absolute_probability=0.0,
        apply_channel_selection_in_training=False, max_masks=256,
        unique_fragment_mode=True, max_enumerated_masks=1024,
        min_mask_relative_probability=0.0, min_fragment_relative_probability=0.0,
        include_single_bond_cuts=False, charge_pooling="logsumexp",
        use_smarts_templates=False,
    )


def probe_candidate_skew(node_dim: int, edge_dim: int, batch) -> None:
    _banner("PROBE 5: train versus eval candidate-set agreement")
    print("  Jaccard between the m/z set selected in train mode and in eval mode.")
    print("  BatchNorm statistics converged on 150 varied 64-molecule batches.")

    def candidate_sets(model, inputs, seed):
        th.manual_seed(seed)
        output = model(**inputs)
        mzs = output["pred_mzs"].detach()
        indices = output["pred_batch_idxs"].detach()
        return {
            int(b): {round(float(m), 4) for m in mzs[indices == b]}
            for b in indices.unique().tolist()
        }

    def jaccard(a: set, b: set) -> float:
        return len(a & b) / max(len(a | b), 1)

    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0]),
        "spec_nce": th.tensor([35.0, 35.0]),
    }
    print()
    for normalization in ("batch", "layer", "graph", "none"):
        for dropout in (0.0, 0.1):
            model = _build_probe_model(dropout, normalization, node_dim, edge_dim)
            encoders = [
                module
                for module in (model.precursor_encoder, model.proposal_encoder)
                if module is not None
            ]
            for encoder in encoders:
                encoder.train()
            with th.no_grad():
                for seed in range(150):
                    sizes = th.randint(
                        8, 60, (64,), generator=th.Generator().manual_seed(seed)
                    ).tolist()
                    warm = _block_diagonal_batch(sizes, node_dim, edge_dim, seed=seed)
                    for encoder in encoders:
                        encoder(*warm)
            model.eval()
            with th.no_grad():
                reference = candidate_sets(model, inputs, 100)
            model.train()
            with th.no_grad():
                draws = [candidate_sets(model, inputs, seed) for seed in range(10)]
            summary = "  ".join(
                f"mol{b} J={sum(jaccard(reference[b], d[b]) for d in draws) / len(draws):.3f}"
                for b in sorted(reference)
            )
            print(f"  norm={normalization:6s} dropout={dropout:<4g}  {summary}")
        print()


def probe_scheduler_wiring() -> None:
    _banner("PROBE 6: does `lr_schedule_type: cosine` reach the scheduler?")
    print("  Config asks for cosine, lr_total_steps 100000, lr_min_ratio 0.01.")
    # Exactly how SpectrumPL.configure_optimizers calls it.
    spectrum_opt = th.optim.AdamW([nn.Parameter(th.zeros(1))], lr=1e-3)
    spectrum_scheduler = build_lr_scheduler(
        optimizer=spectrum_opt, decay_rate=0.95, warmup_steps=1000, decay_steps=5000
    )
    # Exactly how FragnnetPL.configure_optimizers calls it.
    fragnnet_opt = th.optim.AdamW([nn.Parameter(th.zeros(1))], lr=1e-3)
    fragnnet_scheduler = build_lr_scheduler(
        optimizer=fragnnet_opt, decay_rate=0.95, warmup_steps=1000, decay_steps=5000,
        schedule_type="cosine", total_steps=100_000, min_lr_ratio=0.01,
    )
    checkpoints = (0, 1000, 10_000, 25_000, 50_000, 75_000, 100_000)
    print(f"  {'step':>8}  {'SpectrumPL (MASSter)':>24}  {'FragnnetPL (cosine)':>22}")
    for step in range(100_001):
        if step in checkpoints:
            print(
                f"  {step:8d}  {spectrum_opt.param_groups[0]['lr']:24.3e}"
                f"  {fragnnet_opt.param_groups[0]['lr']:22.3e}"
            )
        spectrum_scheduler.step()
        fragnnet_scheduler.step()
    print("  MASSter drops schedule_type/total_steps/min_lr_ratio and gets exponential decay.")


def probe_weight_decay(node_dim: int, edge_dim: int) -> None:
    _banner("PROBE 7: weight decay reaches normalization gains, biases and embeddings")
    model = _build_probe_model(0.1, "batch", node_dim, edge_dim)
    norm_ids, embedding_ids = set(), set()
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            norm_ids.update(id(p) for p in module.parameters(recurse=False))
        if isinstance(module, nn.Embedding):
            embedding_ids.update(id(p) for p in module.parameters(recurse=False))
    total = norm_count = embedding_count = bias_count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        total += parameter.numel()
        if id(parameter) in norm_ids:
            norm_count += parameter.numel()
        elif id(parameter) in embedding_ids:
            embedding_count += parameter.numel()
        elif name.endswith("bias") or parameter.ndim == 1:
            bias_count += parameter.numel()
    excluded = norm_count + embedding_count + bias_count
    print(f"  trainable parameters           : {total:,}")
    print(f"    normalization weight and bias: {norm_count:,}")
    print(f"    embedding tables             : {embedding_count:,}")
    print(f"    remaining one-dimensional    : {bias_count:,}")
    print(f"    conventionally decay-exempt  : {excluded:,} ({100 * excluded / total:.2f}%)")
    print()
    print("  AdamW decouples decay, so it pulls every parameter toward zero each step.")
    print("  Retained fraction if the gradient did not oppose it (lr 1e-3, weight_decay 0.01):")
    for steps in (10_000, 50_000, 100_000, 200_000):
        print(f"    {steps:7,d} steps -> {(1 - 1e-3 * 0.01) ** steps:.3f}")
    print("  lr_schedule is False in 225 of 226 runs, so this pull never decays.")


def probe_wandb_census() -> None:
    _banner("W&B CENSUS: regularization settings actually used")
    import wandb

    api = wandb.Api(timeout=120)
    runs = list(
        api.runs(
            "frag-gnn/frag-gnn",
            filters={"config.model_type": {"$in": [LEGACY_MASSTER_MODEL_TYPE, "masster"]}},
        )
    )
    # Runs from before the rename log `mass_shatter_*` keys; read every run under current names.
    configs = {}
    for run in runs:
        configs[run.id] = dict(run.config)
        migrate_legacy_masster_name(configs[run.id])
    keys = [
        "mol_dropout", "masster_head_dropout",
        "masster_intensity_transformer_dropout", "masster_context_dropout",
        "mol_normalization", "masster_context_normalization",
        "weight_decay", "lr", "lr_schedule", "gradient_clip_val", "train_sample_weight",
    ]
    print(f"  model_type=masster runs: {len(runs)}")
    for key in keys:
        counts = collections.Counter(str(configs[r.id].get(key, "<absent>")) for r in runs)
        rendered = ", ".join(f"{value}x{count}" for value, count in counts.most_common(6))
        print(f"    {key:46s} {rendered}")
    print("\n  Runs whose regularization differs from the corpus default:")
    for run in runs:
        flags = []
        if str(configs[run.id].get("mol_dropout")) != "0.1":
            flags.append(f"mol_dropout={configs[run.id].get('mol_dropout')}")
        if str(configs[run.id].get("masster_head_dropout")) != "0.1":
            flags.append(f"head_dropout={configs[run.id].get('masster_head_dropout')}")
        if str(configs[run.id].get("weight_decay")) != "0.01":
            flags.append(f"weight_decay={configs[run.id].get('weight_decay')}")
        if str(configs[run.id].get("lr")) != "0.001":
            flags.append(f"lr={configs[run.id].get('lr')}")
        if configs[run.id].get("lr_schedule") not in (False, "False"):
            flags.append("lr_schedule=True")
        if configs[run.id].get("mol_normalization") != "batch":
            flags.append(f"mol_normalization={configs[run.id].get('mol_normalization')}")
        if flags:
            print(
                f"    {run.id}  gnn={str(configs[run.id].get('mol_gnn_type')):16s} "
                f"{', '.join(flags)}  {run.name[:56]}"
            )


class _PreLayerNormCandidateLayer(nn.Module):
    """Pre-norm twin of :class:`_InducedCandidateAttentionLayer`, same parameter count.

    Defined here rather than in ``src`` because it exists only as the control arm for
    probe 8; nothing in the model uses it.
    """

    def __init__(self, model_dim: int, num_heads: int, feedforward_dim: int, dropout: float):
        super().__init__()
        self.latent_attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.candidate_attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.latent_attention_norm = nn.LayerNorm(model_dim)
        self.latent_feedforward_norm = nn.LayerNorm(model_dim)
        self.candidate_attention_norm = nn.LayerNorm(model_dim)
        self.candidate_feedforward_norm = nn.LayerNorm(model_dim)
        self.latent_feedforward = bond_fragment_heads._build_feedforward(model_dim, feedforward_dim, dropout)
        self.candidate_feedforward = bond_fragment_heads._build_feedforward(model_dim, feedforward_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, candidates, inducing, padding_mask):
        updates, _ = self.latent_attention(
            self.latent_attention_norm(inducing), candidates, candidates,
            key_padding_mask=padding_mask, need_weights=False,
        )
        inducing = inducing + self.dropout(updates)
        inducing = inducing + self.dropout(
            self.latent_feedforward(self.latent_feedforward_norm(inducing))
        )
        updates, _ = self.candidate_attention(
            self.candidate_attention_norm(candidates), inducing, inducing, need_weights=False
        )
        candidates = candidates + self.dropout(updates)
        candidates = candidates + self.dropout(
            self.candidate_feedforward(self.candidate_feedforward_norm(candidates))
        )
        return candidates, inducing


def probe_post_layer_norm_gradients() -> None:
    _banner("PROBE 8: post-LN versus pre-LN gradient profile, candidate-attention stack")
    print("  Per-layer ||grad|| at initialization, layer 1 (input) .. layer N (output).")
    print("  Loss: linear readout to a per-candidate logit, cross-entropy over a molecule.")
    print("  A scale-only loss such as x.pow(2) is invalid here: post-LN ends in a LayerNorm,")
    print("  so that loss is nearly scale-invariant upstream and fabricates a huge ratio.")

    def profile(layer_cls, depth, model_dim=384, final_norm=False):
        th.manual_seed(5)
        layers = nn.ModuleList([layer_cls(model_dim, 8, 1024, 0.0) for _ in range(depth)])
        readout = nn.Linear(model_dim, 1)
        tail = nn.LayerNorm(model_dim) if final_norm else nn.Identity()
        tokens = th.randn(4, 256, model_dim)
        inducing = th.randn(1, 128, model_dim).expand(4, -1, -1).contiguous()
        mask = th.zeros(4, 256, dtype=th.bool)
        target = th.randint(0, 256, (4,))
        candidates, latent = tokens, inducing
        for layer in layers:
            candidates, latent = layer(candidates, latent, mask)
        logits = readout(tail(candidates)).squeeze(-1)
        th.nn.functional.cross_entropy(logits, target).backward()
        return [
            sum(p.grad.norm().item() ** 2 for p in layer.parameters() if p.grad is not None) ** 0.5
            for layer in layers
        ]

    print()
    for depth in (2, 3, 6, 12):
        post = profile(bond_fragment_heads._InducedCandidateAttentionLayer, depth)
        pre = profile(_PreLayerNormCandidateLayer, depth, final_norm=True)
        print(f"  depth {depth:2d}  post-LN max/min = {max(post) / min(post):.2f}"
              f"   pre-LN max/min = {max(pre) / min(pre):.2f}")
    print("  Production depth is 2 (90 configs) or 3 (56 configs): no post-LN pathology.")


def probe_head_normalization(node_dim: int, edge_dim: int) -> None:
    _banner("PROBE 9: normalization outside the GNN, and bond-head residual attenuation")
    model = _build_probe_model(0.1, "batch", node_dim, edge_dim)
    total = sum(p.numel() for p in model.parameters())
    norm_types = (nn.LayerNorm, nn.BatchNorm1d)
    print(f"  {'module':26s} {'params':>12} {'share':>7} {'norms':>6} {'dropouts':>9}")
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        if not count:
            continue
        norms = sum(
            1 for m in module.modules()
            if isinstance(m, norm_types) or type(m).__name__ == "GraphNorm"
        )
        drops = sum(1 for m in module.modules() if isinstance(m, (nn.Dropout,)))
        drops += sum(1 for m in module.modules() if isinstance(m, nn.MultiheadAttention))
        print(f"  {name:26s} {count:12,} {100 * count / total:6.1f}% {norms:6d} {drops:9d}")

    print("\n  Bond head residual is norm(x + f(x)); the GNN is norm(f(x)) + x.")
    print("  ||branch|| / ||residual|| per layer, by input scale:")
    head = BondFocusedBreakHead(
        atom_embedding_dim=384, bond_feature_dim=9, context_dim=128, bond_state_dim=384,
        head_hidden_dim=512, num_layers=3, dropout=0.0,
    ).eval()
    num_atoms, num_bonds = 600, 650
    endpoints = th.stack([
        th.randint(0, num_atoms, (num_bonds,)), th.randint(0, num_atoms, (num_bonds,))
    ])
    for scale in (0.1, 1.0, 10.0):
        atoms = th.randn(num_atoms, 384) * scale
        context = th.randn(num_bonds, 128) * scale
        with th.no_grad():
            atom_states = head.atom_project(atoms)
            bond_states = head.bond_project(th.cat([
                atoms[endpoints[0]] + atoms[endpoints[1]],
                th.abs(atoms[endpoints[0]] - atoms[endpoints[1]]),
                th.randn(num_bonds, 9), context,
            ], dim=-1))
            flat = endpoints.flatten()
            counts = atom_states.new_zeros(num_atoms).index_add(
                0, flat, atom_states.new_ones(2 * num_bonds)
            )
            ratios = []
            for atom_update, bond_update, atom_norm, bond_norm in zip(
                head.atom_updates, head.bond_updates, head.atom_norms, head.bond_norms
            ):
                messages = atom_states.new_zeros(atom_states.shape).index_add(
                    0, flat, th.cat([bond_states, bond_states], dim=0)
                ) / counts.clamp(min=1).unsqueeze(1)
                branch = atom_update(th.cat([atom_states, messages], dim=1))
                ratios.append((branch.norm() / atom_states.norm()).item())
                atom_states = atom_norm(atom_states + branch)
                left, right = atom_states[endpoints[0]], atom_states[endpoints[1]]
                bond_states = bond_norm(bond_states + bond_update(th.cat(
                    [bond_states, left + right, th.abs(left - right)], dim=1
                )))
        print(f"    input scale {scale:<5g}: {[f'{v:.3f}' for v in ratios]}")
    print("  Stable across scales, unlike the GNN: the bond head can attenuate a layer.")


def probe_context_scale() -> None:
    _banner("PROBE 10: does the context vector's scale vary with the CE-reading count?")
    print("  masster_model.py:603 claims it does. Both CE paths mean-pool, so test it.")
    for normalization in ("none", "layer"):
        th.manual_seed(17)
        encoder = MassterContextEncoder(
            num_precursor_types=4, num_instrument_types=3, num_fragmentation_modes=2,
            use_nce=True, use_ace=False, int_embedder="abs-sines", ce_insert_type="embed",
            ce_insert_size=6, metadata_embedding_dim=32, context_dim=128,
            nce_max=200.0, nce_mean=60.0, nce_std=40.0,
            ace_max=100.0, ace_mean=20.0, ace_std=15.0, normalization=normalization,
        ).eval()
        print(f"\n  context_normalization={normalization}")
        for readings in (1, 4, 22, 86):
            th.manual_seed(2)
            nce = th.rand(readings * 8) * 120 + 10
            indices = th.repeat_interleave(th.arange(8), readings)
            with th.no_grad():
                context = encoder(
                    batch_size=8, precursor_types=th.zeros(8, dtype=th.long),
                    instrument_types=th.zeros(8, dtype=th.long),
                    fragmentation_modes=th.zeros(8, dtype=th.long),
                    nce=nce, nce_batch_indices=indices, ace=None, ace_batch_indices=None,
                )
            norms = context.norm(dim=1)
            print(f"    {readings:3d} readings/spectrum: mean||ctx||={norms.mean():.4f}"
                  f"   across-spectrum sd={norms.std():.4f}")
    print("\n  Magnitude is flat; the across-spectrum spread shrinks with the reading count,")
    print("  which is mean-pooling, not a scale problem. LayerNorm fixes magnitude only.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb", action="store_true", help="also run the W&B run census")
    args = parser.parse_args()

    node_dim, edge_dim = get_mol_feats_sizes(NODE_FEATS, EDGE_FEATS, 0)
    print(f"production molecular feature dims: node={node_dim} edge={edge_dim}")

    probe_dropout_inventory(node_dim, edge_dim)
    probe_dropout_placement()
    probe_residual_attenuation(node_dim, edge_dim)
    probe_batch_composition(node_dim, edge_dim)

    # Probe 5 needs the MASSter chemistry fields, which come from the test fixture's
    # graph builder; its feature set is smaller than the production one above.
    sys.path.insert(0, "tests")
    from test_masster_model import _build_batch  # noqa: PLC0415

    batch = _build_batch(PROBE_SMILES)
    probe_candidate_skew(batch.x.shape[1], batch.edge_attr.shape[1], batch)

    probe_scheduler_wiring()
    probe_weight_decay(node_dim, edge_dim)
    probe_post_layer_norm_gradients()
    probe_head_normalization(node_dim, edge_dim)
    probe_context_scale()
    if args.wandb:
        probe_wandb_census()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
