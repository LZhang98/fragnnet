# Pairwise Loss & Sampler Design

> **Status:** REFERENCE &nbsp;·&nbsp; **Updated:** 2026-10-12
>
> Design of the shipped pairwise loss + sampler (`loss/pairwise.py`, `dataset/data_sampler.py`).

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

## Implemented Loss Functions (`src/fragnnet/loss/pairwise.py`)

All functions output a `[B, B]` symmetric (or asymmetric for CE) similarity matrix.

### Binned / Vectorized (fast, GPU-friendly)

| Function | Config key | Notes |
|---|---|---|
| `pairwise_cosine_similarity()` | `pairwise_cosine_loss_weight` | L2-normalize log-probs → dot product; sparse COO trick avoids 150k-dim dense alloc |
| `pairwise_jss_similarity()` | `pairwise_jss_loss_weight` | JSS = 1 − JSD/log(2); computed in log-space for stability |
| `pairwise_cross_entropy()` | `pairwise_cross_entropy_loss_weight` | **Asymmetric** — must fill full [B,B]; missing bins use log(eps) |

### Hungarian / Peak-aligned (removed)

`get_pairwise_jss_sim_hun()` and `get_pairwise_cross_entropy_hun()` matched peaks by m/z before
computing the similarity. Nothing called them, and they were deleted on 2026-10-12 (reorganization
plan Phase 8 step 0); the code is at revision `13b1599d`.

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

---

## Why CE Beats JSD/sqrt for Retrieval — and What It Teaches Us

### The observation

Empirically: a model trained with JSD + sqrt intensity transform achieves **higher**
JSS-sqrt(pred, true_spectrum) than a CE-trained model — better reconstruction by the retrieval
metric itself. Yet the CE-trained model has **better retrieval rank** when evaluated as
JSS-sqrt(real_spectrum, predicted_candidate).

Better reconstruction → worse retrieval. This is not a contradiction; it is a diagnostic.

### The retrieval setup

Retrieval computes:

```
score(candidate) = JSS-sqrt(real_measured_spectrum, model_predicted_spectrum_of_candidate)
```

The real spectrum is fixed. The model predicts spectra for each candidate molecule. Rank is
determined by:

```
gap = JSS-sqrt(real, q_correct) − max_i JSS-sqrt(real, q_wrong_i)
```

JSD/sqrt improves the first term but inflates the second more, shrinking the gap.

### Why CE's gradient asymmetry keeps predictions molecule-specific

CE loss: `L = −∑ᵢ pᵢ · log(qᵢ)`

Gradient w.r.t. predicted fragment `qᵢ`:

```
∂L/∂qᵢ = −pᵢ / qᵢ    if pᵢ > 0  (true peak)
∂L/∂qᵢ = 0            if pᵢ = 0  (not a true peak)
```

Two properties act together:

**Property 1 — zero gradient for non-true fragments.** The model receives no instruction to
suppress fragments that don't match true peaks. Non-true DAG fragments are suppressed only
implicitly, as a side effect of the softmax normalization pulling mass toward the true peaks.
There is no competing signal that fights the generative structure of the DAG.

**Property 2 — gradient proportional to peak intensity.** A dominant fragment with `pᵢ = 0.80`
generates 16× more gradient than a minor fragment with `pᵢ = 0.05`. The optimizer commits most
of its capacity to getting the large, distinctive peaks right. Common, low-intensity fragments
(water loss, generic neutral losses) receive small gradient and stay low-mass in predictions.

The result: each molecule's CE-trained prediction is a sharp distribution concentrated on that
molecule's high-intensity, structurally distinctive fragments.

### Why JSD/sqrt breaks this

JSD gradient w.r.t. `qᵢ`: `0.5 · log(qᵢ / Mᵢ)` where `M = 0.5·(p + q)`.

- **Not intensity-proportional.** High- and low-intensity peaks receive similar gradient magnitude.
  The model treats a 1% fragment and a 80% fragment as nearly equally important.
- **Explicit suppression of non-true peaks.** When `pᵢ = 0, qᵢ > 0`: gradient = `+0.5·log(2)`,
  actively pushing `qᵢ` down. This competes with the DAG generative objective.
- **sqrt flattens the training target.** A dominant peak at raw intensity 0.9 becomes ≈ 0.51 of
  the mass after sqrt + L1-normalize; a small peak at 0.05 becomes ≈ 0.14. The model is trained
  to predict a flatter distribution where small peaks matter almost as much as large ones.

Net effect: predictions spread mass more uniformly across DAG fragments, including low-intensity
common fragments.

### Why inflated wrong-candidate scores follow

Common fragments (water loss, ring-retaining losses, basic neutral losses) appear in many
molecules' DAGs because BFS fragmentation generates them universally. They also appear in most
real spectra at moderate intensity.

Under CE: a wrong candidate's prediction commits mass to its own distinctive high-intensity
fragments. Those fragments don't appear in the query molecule's real spectrum → near-zero
JSS-sqrt overlap.

Under JSD/sqrt: a wrong candidate's prediction spreads mass to common fragments (amplified by
sqrt in its training target). Those common fragments DO appear in the query's real spectrum →
non-trivial JSS-sqrt overlap even for structurally distinct wrong candidates.

The correct candidate's score improves under JSD/sqrt (confirmed empirically), but wrong
candidate scores inflate more, eroding the rank gap.

### Key assumption to validate

The chain "diffuse predictions → common fragment inflation → worse rank gap" rests on an
assumption: **common fragments tend to be low-intensity in true spectra**. This holds generally
(dominant fragmentation pathways are usually molecule-specific) but not universally (e.g. lipids
with strong common neutral losses). The mechanism can be confirmed directly by computing
JSS-sqrt(real_A, q_B) for a set of wrong candidates B under both training regimes and checking
whether the score distribution shifts.

### Implication for loss design

The observation points to a specific contrastive objective: directly penalize wrong-candidate
inflation during training.

The needed pairwise matrix is **real-spectrum vs. predicted-candidate** (cross-matrix), not the
existing predicted-vs-predicted matrices:

```
S_ij = JSS-sqrt(p_i_real, q_j_pred)   ← what retrieval actually measures
```

Apply a contrastive loss over this matrix (maximize diagonal, minimize off-diagonal). This
directly targets the failure mode: wrong candidates scoring high against real spectra.

CE should remain the primary loss (for gradient asymmetry and sharpness). The contrastive term
adds explicit pressure on the rank gap without sacrificing the molecule-specificity that CE
provides. JSD/sqrt can be kept as a weak secondary term to recover reconstruction accuracy on
the correct candidate, since under CE's dominant gradient it cannot flatten predictions enough
to hurt discrimination.
