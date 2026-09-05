# Fine-tuning FraGNNet

> **Status:** GUIDE &nbsp;·&nbsp; **Updated:** 2026-04-21
>
> How to fine-tune a pretrained checkpoint.

This document describes how to fine-tune pre-trained FraGNNet models for downstream tasks such as spectrum-to-molecule ranking, retrieval, and other MS/MS analysis workflows. It covers configuration options, component freezing strategies, training procedures, and best practices.

## Table of Contents

1. [Overview](#overview)
2. [FraGNNet Components](#fragnnet-components)
3. [Fine-tuning Strategies](#fine-tuning-strategies)
4. [Configuration Guide](#configuration-guide)
5. [Example Configs](#example-configs)
6. [Training Commands](#training-commands)
7. [Monitoring & Resuming](#monitoring--resuming)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Fine-tuning allows you to adapt a pre-trained FraGNNet model to your specific dataset or task while leveraging learned molecular and spectral representations. Two main strategies are supported:

- **Freeze Strategy** (`freeze_all_shared: true`): Freeze pre-trained shared components and only train task-specific heads (recommended for small datasets, reduces overfitting)
- **Full Fine-tune Strategy** (`freeze_all_shared: false`): Train all components with differential learning rates (recommended for larger datasets, better adaptation)

---

## FraGNNet Components

The FraGNNet model consists of these trainable components:

### Shared Components (Pre-trained Representation Learning)
These components are shared across all tasks and learned during pre-training:

- `mol_embedder` - Molecule graph GNN encoder (converts molecular structure → embeddings)
- `mol_pool` - Molecule-level pooling (aggregates atom embeddings → molecule embedding)
- `ce_embedder` - Collision energy embedding (encodes CE values)
- `prec_embedder` - Precursor type embedding (encodes [M+H]+, [M-H]-, [M+Na]+, [M+Cl]-)
- `inst_embedder` - Instrument type embedding (encodes FT, QTOF, IT)
- `formula_embedder` - Chemical formula embeddings (encodes element counts)
- `depth_embedder` - Fragmentation depth embeddings (encodes depth 0-4)
- `complement_embedder` - Complement fragment embeddings (encodes H-gain/loss)

### Task-Specific Components (Prediction Heads)
These components are initialized fresh and trained on the target task:

**When using GNN** (`frag_gnn_type: GNN`):
- `frag_embedder` - Fragment graph GNN encoder
- `frag_pool` - Fragment-level pooling
- `cc_interstage` - Cross-component fusion module

**When using NodeMLP** (`frag_gnn_type: NodeMLP`):
- `frag_embedder` - **NOT USED** (NodeMLP skips GNN)
- `frag_pool` - **NOT USED** (No pooling stage)
- `cc_interstage` - **NOT USED** (Direct MLP prediction)

**Always used**:
- `formula_module` - MLPBlocks for formula/fragment prediction
- `node_module` - Optional MLPBlocks for node-level prediction (if `mlp_output_format: node_formula`)
- `oos_module` - Out-of-spectrum prediction module (if `predict_oos: true`)

---

## Fine-tuning Strategies

### Strategy 1: Freeze Shared Components (Recommended for Small Datasets)

**Configuration**: `freeze_all_shared: true`

**Best for**:
- Limited target data (< 10k samples)
- Quick adaptation needed
- Reducing catastrophic forgetting
- Limited computational resources

**What happens**:
1. Shared components are frozen
2. Task-specific heads are trainable
3. Shared components' parameters are **not included** in optimizer
4. `pretrained_weights_lr_ratio` is ignored

**Config**:
```yaml
finetune: true
finetune_pretrained_ckpt: saved_ckpts/mces_nist20_ma_mi_inchikey_m30_mlpc_edd2j4yy_epoch=49.ckpt
finetune_freeze_all_shared: true
finetune_weights_lr_ratio: 0.1  # Ignored when frozen
finetune_shared_components:
  - mol_embedder
  - mol_pool
  - ce_embedder
  - prec_embedder
  - inst_embedder
```

**Example file**: [config/fragnnet_finetuning/nist20_finetuning_freeze.yml](../../config/fragnnet_finetuning/nist20_finetuning_freeze.yml)

### Strategy 2: Full Fine-tune (Recommended for Large Datasets)

**Configuration**: `freeze_all_shared: false`

**Best for**:
- Large target dataset (> 50k samples)
- Task differs significantly from pre-training
- Sufficient GPU memory (24GB+)
- Want to adapt shared representations

**What happens**:
1. All components are trainable
2. Shared components get reduced learning rate: `lr * pretrained_weights_lr_ratio`
3. Task-specific heads get full learning rate
4. Prevents catastrophic forgetting via differential learning rates

**Config**:
```yaml
finetune: true
finetune_pretrained_ckpt: saved_ckpts/mces_nist20_ma_mi_inchikey_m30_mlpc_edd2j4yy_epoch=49.ckpt
finetune_freeze_all_shared: false
finetune_freeze_components: []
finetune_weights_lr_ratio: 0.1  # Shared LR = 0.0004 * 0.1 = 0.00004
finetune_shared_components:
  - mol_embedder
  - mol_pool
  - ce_embedder
  - prec_embedder
  - inst_embedder
```

**Example file**: [config/fragnnet_finetuning/nist20_finetuning.yml](../../config/fragnnet_finetuning/nist20_finetuning.yml)

---

## Configuration Guide

### Finetuning Configuration Block

```yaml
# ============================================================================
# Finetune Configuration
# ============================================================================
finetune: true                                      # Enable fine-tuning
finetune_pretrained_ckpt: saved_ckpts/mces_nist20_ma_mi_inchikey_m30_mlpc_edd2j4yy_epoch=49.ckpt
finetune_strict_load: True                          # Strict checkpoint loading
finetune_freeze_all_shared: true                    # true = freeze shared; false = fine-tune all
finetune_freeze_components:                         # Optional, ignored if freeze_all_shared is set
  - mol_embedder      # Molecule GNN encoder
  - mol_pool          # Molecule pooling
finetune_shared_all: false                          # Whether to share all components
finetune_shared_components:                         # Components to freeze/fine-tune
  - mol_embedder      # Molecule GNN encoder
  - mol_pool          # Molecule pooling
  - ce_embedder       # Collision energy embedding
  - prec_embedder     # Precursor type embedding
  - inst_embedder     # Instrument type embedding
finetune_weights_lr_ratio: 0.1                      # Shared LR = main_lr × 0.1 (only if freeze_all_shared: false)
```

`finetune_strict_load: True` requires matching keys and tensor shapes within the
selected shared components, and rejects mismatches before copying any weights.
Use `finetune_strict_load: False` for intentional partial transfers: missing or
shape-incompatible tensors remain initialized, incompatible shapes are logged,
and only successfully copied tensors receive the shared learning rate or freezing.
Corrected `node_formula` input dimensions may require this partial-transfer mode
for checkpoints with an incompatible node head. Its `ce_scale` and `depth_ce_bias`
settings now affect predictions when enabled; they were previously ignored.

### Data Configuration

```yaml
# Data paths - CHANGE these for your target dataset
spec_fp: data/proc/[YOUR_DATASET]/spec_df.pkl        # Spectrum dataframe
mol_fp: data/proc/[YOUR_DATASET]/mol_df.pkl          # Molecule dataframe
split_dp: data/split/[YOUR_DATASET]_split            # Train/val/test split
frag_dp: data/frag/[YOUR_DATASET]_d3_h4_isoFalse/dags # Fragment DAGs
mces_fp: data/proc/mces/mces_pairs.pkl                # Optional MCES pairs
```

### Model Architecture

```yaml
num_depth: 4                 # Fragmentation depth (0-4)
num_hs: 4                    # Max hydrogen count (±4)
frag_gnn_type: NodeMLP       # Fragment encoder type: "NodeMLP" or "GNN"

frag_params:
  pyg_edge_feats: []         # No edge features
  pyg_edges: False           # Don't use PyG edge_index
  pyg_bigraph: False         # Not a bipartite graph
  preload: True              # Preload features
  preprocess: True           # Preprocess on startup

spec_params:
  merge: False
  nce: True                  # Normalize collision energy
  merge_keep_ces: False
  preprocess: True
  prec_type_str: False
  prec_types:                # Precursor ion types
    - "[M+H]+"
    - "[M-H]-"
    - "[M+Na]+"
    - "[M+Cl]-"
  inst_type: True
  inst_types:                # Instrument types
    - "FT"
    - "QTOF"
    - "IT"

mol_params:
  preprocess: True
```

### Feature Insertion Configuration

```yaml
# Where to insert context features (CE, precursor, instrument)
# Options: "none", "mol" (molecule level), "mlp" (after fragment encoding), "frag" (unsupported)

ce_insert_type: embed                # How to embed CE: "id", "lin", "embed", "bin"
ce_insert_location: mlp              # Insert at MLP level (after frag encoding)
ce_insert_merge: False               # Don't merge with other embeddings
ce_insert_size: 32                   # CE embedding dimension

prec_insert_location: mlp            # Precursor at MLP level
prec_insert_size: 32                 # Precursor embedding dimension

inst_insert_location: mlp            # Instrument at MLP level
inst_insert_size: 32                 # Instrument embedding dimension
```

### Training Hyperparameters

```yaml
# Learning rate and optimization
lr: 0.0004                          # Main learning rate (typical range: 0.0001-0.001)
lr_schedule: False                  # Use learning rate scheduling

# Batch sizes
train_batch_size: 256               # Adjust based on GPU memory
eval_batch_size: 256
accumulate_grad_batches: 1          # Accumulate gradients (useful for smaller batch sizes)

# Training duration
min_epochs: 100                     # Minimum epochs before early stopping
max_epochs: 100                     # Maximum training epochs

# Gradient settings
gradient_clip_val: 5.0              # Clip gradients to prevent explosion

# Precision and compilation
precision: 32                       # 32-bit (default) or 16 (mixed precision)
use_tensor_float32: True            # Use TF32 on Ampere GPUs for speedup
automatic_optimization: False       # Manual optimization (advanced)
compile: False                       # Torch compile for performance
```

### Loss and Evaluation Configuration

```yaml
debug_overfit: False                # Debug mode (overfit on small batch)
mask_null_formula: False            # Mask NULL formulae in loss
predict_oos: True                   # Predict out-of-spectrum peaks
oos_tolerance_multiple: 1           # OOS tolerance (1x relative abundance)
oos_loss: True                      # Include OOS in loss

# Spectrum representation
gaussian_renormalize: True          # Renormalize with Gaussian convolution
mz_bin_res: 0.01                    # M/Z bin resolution (0.01 = 10 ppm @ mz 1000)
mz_max: 1500.0                      # Maximum m/z value

sparse_cosine_similarity: True      # Use sparse cosine similarity
mol_pool_type: mean                 # Molecule pooling: "mean", "sum", "mean_max_sum"
frag_pool_type: mean                # Fragment pooling: "mean", "sum", "mean_max_sum"
```

### Checkpointing and Logging

```yaml
upload_checkpoints: True             # Upload to W&B
checkpoint_metric: val_cos_sim_0.01_epoch/mean  # Metric to track
auxiliary_scores:                   # Additional metrics
  - cos_sim
eval_mz_bin_res:                    # Evaluation resolutions
  - 0.01

wandb_name: nist20_finetuning_freeze  # W&B run name
wandb_tags:                         # W&B tags for filtering
  - finetuning
  - nist20
  - freeze_shared
```

### Data Processing

```yaml
pin_memory: True                    # Pin GPU memory for faster data loading
loss_vectorized: True               # Vectorized loss computation
loss_batch_size: 16                 # Batch size for loss computation
             # Skip edge and other auxiliary losses

dynamic_batch_sampler: SpecMolFragDynamicBatchSampler  # Dynamic batching
pl_enable_progress_bar: True        # Progress bar
activation_checkpointing: False     # Gradient checkpointing
```

---

## Example Configs

### Example 1: Freeze Strategy (nist20_finetuning_freeze.yml)

**Use case**: Small dataset, quick adaptation, NodeMLP architecture

```yaml
wandb_name: nist20_finetuning_freeze
wandb_tags:
  - finetuning
  - nist20
  - freeze_shared

# Data
spec_fp: data/proc/nist20/spec_df.pkl
mol_fp: data/proc/nist20/mol_df.pkl
split_dp: data/split/nist20_ma_mi_inchikey
frag_dp: data/frag/nist20_ma_mi_d3_h4_isoFalse/dags
mces_fp: data/proc/mces/mces_pairs.pkl

# Freeze shared components
finetune: true
finetune_pretrained_ckpt: saved_ckpts/mces_nist20_ma_mi_inchikey_m30_mlpc_edd2j4yy_epoch=49.ckpt
finetune_freeze_all_shared: true
finetune_weights_lr_ratio: 0.1

# Model - NodeMLP (no frag_embedder or frag_pool)
num_depth: 4
num_hs: 4
frag_gnn_type: NodeMLP

# Training
train_batch_size: 256
eval_batch_size: 256
lr: 0.0004
max_epochs: 100
gradient_clip_val: 5.0

# Loss
predict_oos: True
oos_loss: True
```

### Example 2: Full Fine-tune Strategy (nist20_finetuning.yml)

**Use case**: Large dataset, task-specific adaptation, NodeMLP architecture

```yaml
wandb_name: nist20_mces_finetuning_m30_mlpc
wandb_tags:
  - finetuning
  - freeze_false
  - nist20

# Data
spec_fp: data/proc/nist20/spec_df.pkl
mol_fp: data/proc/nist20/mol_df.pkl
split_dp: data/split/nist20_ma_mi_inchikey
frag_dp: data/frag/nist20_ma_mi_d3_h4_isoFalse/dags
mces_fp: data/proc/mces/mces_pairs.pkl

# Fine-tune all components with differential LR
finetune: true
finetune_pretrained_ckpt: saved_ckpts/mces_nist20_ma_mi_inchikey_m30_mlpc_edd2j4yy_epoch=49.ckpt
finetune_freeze_all_shared: false                    # All trainable
finetune_freeze_components: []                       # Override template default
finetune_weights_lr_ratio: 0.1                      # Shared: 0.00004

# Model - NodeMLP
num_depth: 4
num_hs: 4
frag_gnn_type: NodeMLP

# Training
train_batch_size: 256
eval_batch_size: 256
lr: 0.0004                # Shared: 0.0004 * 0.1 = 0.00004
max_epochs: 100
gradient_clip_val: 5.0

# Loss
predict_oos: True
oos_loss: True
```

---

## Training Commands

### Local GPU Training

```bash
# Freeze strategy - best for quick fine-tuning
python scripts/run_pl_model_fit.py \
  --custom_fp config/fragnnet_finetuning/nist20_finetuning_freeze.yml

# Full fine-tune strategy - best for larger datasets
python scripts/run_pl_model_fit.py \
  --custom_fp config/fragnnet_finetuning/nist20_finetuning.yml
```

### SLURM Cluster Training

Create `slurm_scripts/fraggnn_finetune.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=fraggnn_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G

module load cuda/11.8
source venv/bin/activate

python scripts/run_pl_model_fit.py \
  --custom_fp config/fragnnet_finetuning/nist20_finetuning_freeze.yml
```

Submit:
```bash
sbatch slurm_scripts/fraggnn_finetune.sh
```

### Resume Training with W&B

Run with a stable job ID and online W&B logging. Re-running the same command resumes from
`last.ckpt` when the earlier run and checkpoint are available:

```bash
python scripts/run_pl_model_fit.py \
  --custom_fp config/fragnnet_finetuning/nist20_finetuning_freeze.yml \
  --wandb_mode online \
  --job_id nist20_finetuning_freeze
```

---

## Monitoring & Resuming

### W&B Monitoring

Key metrics to track:

- **Loss**: `train_loss`, `val_loss` (should decrease smoothly)
- **Cosine Similarity**: `val_cos_sim_0.01_epoch/mean` (checkpoint metric)
- **Learning Rate**: `learning_rate` (verify correct per-component rates)
- **Spectrum Entropy**: `val_spectrum_entropy_epoch/mean` (prediction quality)

### Checkpointing

Checkpoints are saved based on:
```yaml
checkpoint_metric: val_cos_sim_0.01_epoch/mean  # Best checkpoint criterion
upload_checkpoints: True                         # Upload to W&B
```

Best checkpoint is automatically selected for inference.

---

## Troubleshooting

### Issue: Loss not decreasing from Epoch 1

**Possible causes**:
- Learning rate too high
- Components not properly frozen
- Checkpoint not loaded correctly

**Solutions**:
```yaml
# 1. Verify freeze is working
finetune_freeze_all_shared: true

# 2. Reduce learning rate
lr: 0.0001  # Try 10x smaller

# 3. Enable gradient clipping
gradient_clip_val: 1.0
```

### Issue: Out of Memory (OOM)

**Solution**:
```yaml
train_batch_size: 128              # Reduce from 256
accumulate_grad_batches: 2         # Accumulate over 2 batches
eval_batch_size: 128
```

### Issue: Model Overfitting (Large train-val gap)

**Solution** (use freeze strategy):
```yaml
finetune_freeze_all_shared: true          # Freeze shared components
lr: 0.0001                                # Reduce LR further
max_epochs: 50                            # Fewer epochs
train_batch_size: 256                    # Keep batch size large
```

### Issue: Model Underfitting (Both losses high)

**Solution** (use full fine-tune strategy):
```yaml
finetune_freeze_all_shared: false         # Allow shared adaptation
finetune_freeze_components: []            # Do not retain the template's component freeze
finetune_weights_lr_ratio: 0.1
lr: 0.0004                                # Standard LR
max_epochs: 100                           # More epochs
```

---

## Quick Reference: Freeze vs. Full Fine-tune

| Aspect | Freeze (`true`) | Full Fine-tune (`false`) |
|--------|-----------------|-------------------------|
| **Best for** | Small datasets (< 10k) | Large datasets (> 50k) |
| **Shared comp. LR** | Frozen (0.0) | Reduced (0.1 × main_lr) |
| **Task comp. LR** | Full (`lr`) | Full (`lr`) |
| **Overfitting risk** | Low | Medium |
| **GPU memory** | Low | High |
| **Training time** | Fast | Slower |
| **Adaptation** | Limited | Strong |
| **Config file** | [nist20_finetuning_freeze.yml](../../config/fragnnet_finetuning/nist20_finetuning_freeze.yml) | [nist20_finetuning.yml](../../config/fragnnet_finetuning/nist20_finetuning.yml) |

---

## NodeMLP Architecture Note

Both example configs use `frag_gnn_type: NodeMLP`, which is a simplified architecture that:
- Skips GNN-based fragment encoding
- Directly predicts fragment properties with MLPBlocks
- Results in **fewer parameters** and **faster training**
- Suitable for quick fine-tuning with limited data

When using NodeMLP, these components are **not used**:
- `frag_embedder` (GNN encoder)
- `frag_pool` (pooling stage)
- `cc_interstage` (cross-component fusion)

This makes NodeMLP ideal for fine-tuning scenarios with resource constraints.

---

## Recommended Fine-tuning Workflow

1. **Start with Freeze Strategy**
   ```bash
   python scripts/run_pl_model_fit.py \
     --custom_fp config/fragnnet_finetuning/nist20_finetuning_freeze.yml
   ```
   - Quick baseline
   - Observe convergence
   - Monitor `val_cos_sim_0.01_epoch/mean`

2. **Evaluate Results**
   - If overfitting: Keep freeze strategy ✓
   - If underfitting: Switch to full fine-tune

3. **Full Fine-tune (if needed)**
   ```bash
   python scripts/run_pl_model_fit.py \
     --custom_fp config/fragnnet_finetuning/nist20_finetuning.yml
   ```
   - Use differential learning rates
   - Monitor per-component gradients

4. **Save Best Checkpoint**
   - W&B automatically selects best by `checkpoint_metric`
   - Download from `saved_ckpts/`

---

## Additional Resources

- **Main Model**: [src/fragnnet/model/fragnnet_model.py](../../src/fragnnet/model/fragnnet_model.py)
- **Data Preparation**: [preproc_scripts/](../../preproc_scripts/)
- **Inference**: [docs/guides/run_inference.md](run_inference.md)
- **Training Script**: [scripts/run_pl_model_fit.py](../../scripts/run_pl_model_fit.py)
