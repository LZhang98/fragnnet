# Pairwise Loss & Sampler Design

## Overview

Pairwise loss trains the model such that the **relative similarities between predicted spectra** match
the **relative similarities between true spectra** within a mini-batch. Concretely, for a batch of B
spectra:

1. Compute a `[B, B]` pairwise similarity matrix from the **true** spectra.
2. Compute the same `[B, B]` matrix from the **predicted** spectra.
3. Minimize MSE between the two matrices.

The model is therefore penalized when it produces spectra whose mutual similarities differ from the
ground-truth pattern — encouraging consistent relative structure across predictions.

---

## Implemented Loss Functions (`src/fragnnet/model/loss.py`)

All functions output a `[B, B]` symmetric (or asymmetric for CE) similarity matrix.

### Binned / Vectorized (fast, GPU-friendly)

| Function | Config key | Notes |
|---|---|---|
| `get_pairwise_cossim()` | `pairwise_cosine_loss_weight` | L2-normalize log-probs → dot product; sparse COO trick avoids 150k-dim dense alloc |
| `get_pairwise_jss_sim()` | `pairwise_jss_loss_weight` | JSS = 1 − JSD/log(2); computed in log-space for stability |
| `get_pairwise_cross_entropy()` | `pairwise_cross_entropy_loss_weight` | **Asymmetric** — must fill full [B,B]; missing bins use log(eps) |

### Hungarian / Peak-aligned (slower, more accurate)

| Function | Notes |
|---|---|
| `get_pairwise_jss_sim_hun()` | m/z-based peak matching before JSS computation |
| `get_pairwise_cross_entropy_hun()` | m/z-based peak matching before CE computation |

---

## Implemented Samplers (`src/fragnnet/dataset/data_sampler.py`)

### Index samplers (yield per-epoch sample indices)

| Class | Config key | Grouping key | Purpose |
|---|---|---|---|
| `WeightedRandomSampler` (via `get_group_sampler`) | `simple_group_sampler: True` | `group_id` or `mol_id` | Weight samples by 1/group-size or 1/mol-size; reduces over-represented molecules |
| `GroupSampler` | `group_sampler: True` | `group_id` | Sample exactly K spectra per group per epoch; epoch-aware shuffle |
| `FormulaPrecTypeGroupSampler` | `formula_prec_type_group_sampler: True` | `prec_type::formula` | Sample K indices per (adduct, formula) composite key — enables hard-negative batches |

### Batch sampler (wraps index sampler)

| Class | Config key | Purpose |
|---|---|---|
| `SpecMolFragDynamicBatchSampler` | `dynamic_batch_sampler: True` | Fills batches up to `max_num` frag nodes/edges; prevents OOM on variable-complexity molecules |
| `DualGroupDynamicBatchSampler` | `dual_group_dynamic_batch_sampler: True` | Fills batches with both mol-level AND formula-level diversity (two group dicts) |

---

## Existing Configs (`config/pairwise_loss/`)

| Config | Loss (weight) | Sampler |
|---|---|---|
| `pairwise_cos.yml` | cosine (0.5) | `dynamic_batch_sampler` + `simple_group_sampler` |
| `pairwise_cos2.yml` | cosine (0.5) | `dynamic_batch_sampler` only |
| `pairwise_jss.yml` | JSS (0.5) | `dynamic_batch_sampler` |
| `pairwise_ce.yml` | CE (0.5) | `dynamic_batch_sampler` + `simple_group_sampler` |

**None of the existing configs use `formula_prec_type_group_sampler`.**

---

## Why Current Pairwise Loss Setup Is Not Ready

The pairwise loss is most useful when a mini-batch contains **structurally related but distinct
molecules** — so that the `[B, B]` similarity matrix has meaningful off-diagonal structure for the
model to learn from.

With the current samplers (`simple_group_sampler` / `dynamic_batch_sampler`), batches are assembled
by group or molecule frequency weighting, with no guarantee that any two spectra in the batch share
the same precursor formula. The resulting similarity matrices are dominated by near-zero
off-diagonal entries (unrelated molecules), providing a weak training signal.

### What's needed: `FormulaPrecTypeGroupSampler` + pairwise loss

The `FormulaPrecTypeGroupSampler` groups samples by `prec_type::formula`, so each mini-batch
naturally contains multiple spectra from the same formula (different structures, adducts, or CEs).
This creates informative off-diagonal similarities — molecules with the same formula but different
structures should have partially overlapping spectra, and the model must learn to distinguish them.

```
Current:  batch ~ random molecules → pairwise matrix ~ identity → weak signal
Proposed: batch ~ formula-grouped  → pairwise matrix has real structure → strong hard-negative signal
```

### Plan: new config combining both

```yaml
# proposed: config/pairwise_loss/fraggnn_d3_ma_mi_nist20_pairwise_cos_formula_sampler.yml
formula_prec_type_group_sampler: True
formula_prec_type_group_sampler_max_per_group: 5
dynamic_batch_sampler: True           # still wrap with dynamic to avoid OOM
pairwise_cosine_loss_weight: 0.5
pairwise_jss_loss_weight: 0.0
pairwise_cross_entropy_loss_weight: 0.0
```

Additional open questions before running:

- **Similarity target calibration**: with formula-grouped batches, should the pairwise target be
  cos_sim, JSS, or Tanimoto of structures? The current target is spectrum-vs-spectrum similarity
  (cos_sim on peaks), which may be noisier than a structure-based label.
- **Loss weight tuning**: weight=0.5 was chosen without formula-grouped batches; with more
  informative off-diagonal entries the pairwise loss may need rebalancing relative to the main
  spectrum prediction loss.
- **Formula coverage**: rare formulas (singleton groups) contribute no pairwise signal. Consider
  filtering to `min_group_size >= 2` before sampling, or using `DualGroupDynamicBatchSampler` to
  guarantee both mol-level and formula-level diversity.
- **CE interaction**: pairwise loss operates on merged peak predictions; need to verify behavior
  with multiple CEs per molecule in the same batch (CE conditioning still active during pairwise
  matrix computation).
