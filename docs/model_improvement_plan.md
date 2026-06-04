# Model Improvement Plan

Analysis based on NIST23 test set predictions.

**Two evaluation setups are used throughout this document — numbers are NOT directly comparable between them:**

| Setup | Description | Spectra per molecule | Source |
|---|---|---|---|
| **Per-CE** | Individual spectra at each collision energy | ~27 (multiple CEs per molecule) | 9-fold CV, 764,637 total |
| **Merged** | Peaks from all CEs combined into one spectrum per molecule | 1 | Scaffold split, used in §4, §5, §9 |

Merged spectra yield systematically higher metrics (low-CE + high-CE fragments all present) and should not be compared directly to per-CE results.

---

## Current State

**Overall (764,637 spectra, 28,256 unique molecules — NIST23 9-fold CV, per-CE spectra)**

| Metric | Value |
|---|---|
| Median cosine sim (tol=0.01) | 0.759 |
| Mean cosine sim (tol=0.01) | 0.658 |
| Median cos\_hun | 0.786 |
| Mean cos\_hun | 0.681 |
| cos\_sim ≥ 0.9 | 34.5% |
| cos\_sim ≥ 0.7 | 55.0% |
| cos\_sim ≥ 0.5 | 69.1% |
| cos\_sim ≥ 0.2 | 86.5% |

**By ion mode**

| Ion mode | Median cos\_sim | n |
|---|---|---|
| Positive (P) | 0.764 | 572,107 |
| Negative (N) | 0.739 | 192,525 |

**By dataset source**

| Dataset | Median cos\_sim | n |
|---|---|---|
| nist23\_hr1 | 0.778 | 396,028 |
| nist23\_hr2 | 0.763 | 302,648 |
| mona23 | 0.563 | 65,961 |

---

## What Is Already Done

The current model (`fragnnet_d3_nist23_cv0`) already has these enabled — **these are not action items**:

| Feature | Status | Config key |
|---|---|---|
| NCE conditioning (feature) | ✅ enabled | `nce: True`, `ce_insert_location: mlp` |
| **CE FiLM scaler (output logits)** | ✅ **implemented** | `ce_scale: True`, `ce_scaler_hidden_dim: 64` |
| Instrument type conditioning | ✅ enabled | `inst_type: True`, `inst_insert_location: mlp` |
| Precursor type conditioning | ✅ enabled | `prec_insert_location: mlp` |
| ACE → NCE conversion for QTOF | ✅ done | `fill_missing_nce()` via `ace_to_nce()` |
| FT / QTOF / IT in inst\_types | ✅ included | `inst_types: [FT, QTOF, IT]` |
| **Fragmentation mode conditioning** | ✅ **implemented** | `frag_mode: True`, `frag_mode_insert_location: mlp`, `frag_mode_scale: True` |
| **Dual CE input (ACE + NCE)** | ✅ **implemented** | `use_nce: True`, `use_ace: True`; independent embeddings are concatenated |

The 0.762 median is with the implemented conditioning active. Remaining QTOF/CID work is now
mostly an ablation/data-balance question, not a missing dual-CE architecture path.

---

## Priority Order

| Priority | Area | Expected Impact | Effort |
|---|---|---|---|
| 1 | `force_stopped` + formula-recall (AR) diagnostic per class | Unblocks §4 and §5 decisions | Very low (query) |
| 2 | **Frag-mode conditioning** (HCD/CID token) | High — FT/CID gap 0.785→0.497, model blind to it | Low (arch, same pattern as inst\_type) |
| 3 | QTOF/CID upweighting in sampler | High — sampler may not weight by inst/frag\_mode | Low |
| 3a | **Adduct-type FiLM** (`prec_type`) | Medium — cheap output-level correction for `[M+Na]+`, `[M+Cl]-`, `[M-H]-` | Low |
| 3b | **Categorical stratum FiLM** (`inst_type`, `frag_mode`, adduct) | Medium–High — extends adduct FiLM to full acquisition strata beyond CE FiLM | Low |
| 4 | Chemical class focal loss weighting | Medium — size explains only 4.2% of variance | Low–Medium |
| 4a | **Peak-rank weighted loss** | Improves top-peak fidelity for perceived quality and retrieval | Low |
| 4b | ~~**Neutral-loss auxiliary loss**~~ | ~~Teaches chemically meaningful losses like H₂O, CO₂, NH₃, SO₂~~ — **REJECTED**: NL class is trivially derivable from input formula features; head trains nothing new | — |
| 4c | **Formula-recall-aware weighting** | Separates model failures from DAG coverage failures | Low–Medium |
| 5 | Prioritized BFS for polycyclic DAG truncation | High — anthracenes 0.276, AR=0.81 vs GrAFF 0.95 | Medium (Cython) |
| 5a | **One-shot 2-bond ring cleavage pre-pass** | High — RDA/fused-ring fragments at depth 1, fixes low `wrecall` for steroids/coumarins/indoles | Medium (Cython, ~190 checks/mol) |
| 5b | **DAG mass filter (prune, not expand)** | Medium — SMARTS prepass/MBFS experiments show expansion hurts (−0.007 cos\_sim); mass filter recovers this at near-zero cost | Low (filter at dataset prep) |
| 6 | **Explicit fragmentation rules for fused rings/steroids** | Medium — 3-bond cleavages not covered by one-shot pre-pass | Medium (SMARTS) |
| 7 | CID training data sourcing | High — FT/CID median 0.497 with ~8% of data | Low (data work) |
| 8 | Dual CE input (ACE + NCE, no conversion) ablation | Medium — tests whether avoiding ACE→NCE noise helps QTOF | Low (config + validation run) |
| 9 | MoNA FT-only gap check | Low — 2-line diagnostic | Very low (query) |
| 10 | **Model ensembling** | +4% cos\_hun, no arch change needed | Very low (multi-seed inference) |
| 10a | **Ensemble distillation student** | Recover part of ensemble gain at single-model inference cost | Medium |
| 11 | OOS calibration per stratum | Confidence filter for downstream use | Very low (query) |
| 11a | **Prediction-quality calibration head** | Confidence score for filtering/reranking downstream retrieval | Low–Medium |

---

## 1. Data Imbalancing Plan

### Problem

The training set is heavily dominated along multiple axes simultaneously:

| Axis | Dominant group | Dominant % | Underrepresented groups |
|---|---|---|---|
| Instrument | FT (Orbitrap) | ~94% | QTOF (~6%), IT (~0%) |
| Frag mode | HCD | ~87% | CID (~13%) |
| Adduct | `[M+H]+` | ~66% | `[M+Na]+`, `[M+Cl]-`, `[M-H]-` |
| Ion mode | Positive | ~74% | Negative (~26%, median −0.04 gap) |
| Dataset source | NIST23 | ~91% | MoNA23 (~9%) |
| Chemical class | Benzenoids + organoheterocyclics | ~60% | steroids, lignans, benzopyrans, etc. |

**Adduct performance** (all 9 folds):

| Adduct | Median cos\_sim | n |
|---|---|---|
| `[M+H]+` | 0.768 | 514,124 |
| `[M-H]-` | 0.744 | 183,441 |
| `[M+Na]+` | 0.720 | 57,971 |
| `[M+Cl]-` | 0.626 | 9,092 |

Note: `[M-H]-` has a large count (n=21,078) but a ~3.5% gap vs `[M+H]+` — worth upweighting alongside `[M+Na]+`.

These imbalances compound: a steroid measured on a QTOF instrument with CID and `[M+Na]+` sits in
the tail of every axis simultaneously.

**Key finding:** Class size explains only **4.2% of performance variance** (r²=0.042,
log(n) vs median cos\_sim regression across all classes with n≥50).  Chemistry dominates —
oversampling alone will not fix structurally hard classes.

### Strategy

#### Step 1 — Existing sampler (audited ✅)

Config (`fragnnet_d3_nist23_cv0.yml`): `simple_group_sampler: True`, `type: group`,
`avg_per_group: 5`.

This calls `get_group_sampler()` in `src/fragnnet/dataset/data_sampler.py`, which
creates a `WeightedRandomSampler` with:

- `sample_weights = 1 / spec_per_group` — inverse CE-count weighting per mol+adduct group
- `num_samples = sum(min(spec_per_group, 5) for each group)` — targets ~5 spectra per mol+adduct per epoch
- `replacement=False`

**What it does:** Each mol+adduct combination (= `group_id` in NIST23 unmerged) contributes
~5 randomly-sampled CE measurements per epoch.  Molecules measured at many CEs (20+) are
downweighted so they don't dominate.

**What it does NOT do:** No weighting by `inst_type`, `frag_mode`, adduct, or chemical class.
QTOF/CID spectra are not upweighted — **Priority 3 confirmed needed.**

Keep this sampler as-is.  The QTOF/CID weighting is a multiplicative addition on top.

#### Step 2 — Add QTOF/CID upweighting to sampler

The existing `get_group_sampler()` only weights by CE count.  Add a stratum multiplier
based on `(inst_type, frag_mode)` using inverse-sqrt frequency dampening, applied on top
of the existing `1/spec_per_group` weights.  This is a `data_sampler.py` change only —
no model or config changes required beyond a new config key:

```python
# Multiply existing weight by stratum factor
stratum_key = (inst_type, frag_mode)          # start simple; extend to adduct if needed
stratum_count = count of training spectra per stratum
stratum_weight = 1 / sqrt(stratum_count)      # FT/HCD ~87% → low multiplier; QTOF/CID ~6% → high
sample_weight[i] *= stratum_weight[stratum_key[i]]
```

Config key to add: `simple_group_sampler_inst_frag_upweight: True`

#### Step 3 — Chemical class focal weighting in loss

Size explains almost nothing.  For classes with residual < -0.10 from the size-regression
line, apply a soft focal weight to the per-spectrum loss:

```python
focal_weight = 1.0 + alpha * max(0, global_median - class_median)
```

Suggested `alpha = 2.0`.  Target classes (sorted by size-adjusted residual, excluding
`n_mol=1` singletons like tetracyclines which are a separate CE-curve problem):

| Class | Median | Size-adj residual | n | n\_mol |
|---|---|---|---|---|
| Anthracenes | 0.276 | -0.439 | 507 | 29 |
| 2-arylbenzofuran flavonoids | 0.360 | -0.318 | 124 | 6 |
| Furans | 0.383 | -0.313 | 246 | 16 |
| Morphinans | 0.380 | -0.299 | 124 | 4 |
| Thienodiazepines | 0.334 | -0.322 | 52 | 4 |
| Lactones | 0.416 | -0.281 | 258 | 10 |
| Benzopyrans | 0.511 | -0.221 | 985 | 46 |
| Steroids | 0.595 | -0.152 | 1793 | 90 |
| Prenol lipids | 0.634 | -0.127 | 3101 | 154 |

Note: **tetracyclines** (n=135, n\_mol=1, residual -0.492) is excluded — it is a
single-molecule CE generalisation problem, not a class-level issue.

#### Step 4 — Oversample rare adducts

`[M+Cl]-` (n=988, ~1.2%), `[M+Na]+` (n=6,816, ~8%), and `[M-H]-` are underrepresented
relative to `[M+H]+`.  Apply upweighting multipliers: `[M+Cl]-` 3×, `[M+Na]+` 1.5×,
`[M-H]-` 1.2× (moderate, since it is already large but underperforms).

#### Step 5 — Separate train/val splits per stratum

Ensure the validation set mirrors per-stratum composition so improvements across all
strata are visible in aggregate val metrics.  Current splits dominated by FT/HCD may mask
regressions on QTOF/CID.

### Evaluation Signal

Log the following groups separately to W&B per validation epoch:
- `val/cos_sim_FT_HCD`, `val/cos_sim_QTOF_CID`, `val/cos_sim_FT_CID`
- `val/cos_sim_Mplus_H`, `val/cos_sim_Mplus_Na`, `val/cos_sim_Mplus_Cl`, `val/cos_sim_Mminus_H`
- `val/cos_sim_ion_pos`, `val/cos_sim_ion_neg`
- `val/cos_sim_oos_low` (pred\_oos\_prob ≤ 0.05), `val/cos_sim_oos_high` (pred\_oos\_prob > 0.2)
- Per-class median for the 9 focal-weight target classes above

---

## 1b. Loss and DAG-Aware Training Objectives

These ideas change the training signal rather than the architecture.  They should be
validated after the basic stratum/adduct imbalance fixes, because they can change what
the model optimizes even when aggregate cosine looks similar.

### Formula-Recall-Aware Loss Weighting

If the true annotated formula is absent from the DAG, the neural model cannot assign
probability to it.  Penalizing that case as a normal model error confounds two different
failures:

| Failure mode | Meaning | Fix |
|---|---|---|
| High DAG recall, low spectrum quality | Candidate exists but model scores it poorly | model/loss fix |
| Low DAG recall, low spectrum quality | Candidate missing from DAG | fragmentation/DAG fix |

Use formula recall and weighted formula recall as either diagnostics or sample weights:

```text
wrecall = intensity fraction of annotated true peaks represented by DAG formulas
```

Recommended first step is diagnostic-only:

```python
df.groupby(pd.cut(df.wrecall_h1, [0, .5, .8, .95, 1.0]))["cos_hun"].median()
```

Training-weight options:

1. Downweight impossible examples:

```python
loss_weight = clamp(wrecall, min=0.2, max=1.0)
loss = loss_weight * loss
```

2. Upweight learnable failures:

```python
if wrecall > 0.9 and cos_hun is low:
    increase sample/class weight
```

Use option 1 cautiously: too much downweighting can hide hard chemistry classes that need
DAG work.  The most useful output may be a routing decision: low-recall classes should go
to DAG generation improvements, high-recall classes should go to model/loss improvements.

Suggested config:

```yaml
formula_recall_weighting: false
formula_recall_weight_key: wrecall_0.01_h1
formula_recall_min_weight: 0.2
```

---

## 2. CID vs HCD Plan

### Problem

**Frag-mode marginals** (all 9 folds):

| frag\_mode | Median cos\_sim | n |
|---|---|---|
| HCD | 0.781 | 663,143 |
| CID | 0.547 | 101,494 |

**By (inst\_type, frag\_mode) cross-table** (single-fold estimate — cross-table not yet recomputed for 9-fold):

| Condition | Median cos\_sim | n (single fold) |
|---|---|---|
| FT / HCD | 0.785 | 74,263 |
| QTOF / CID | 0.616 | 4,806 |
| FT / CID | 0.497 | 6,538 |
| IT / CID | 0.000 | 35 |

FT/CID (0.497) is worse than QTOF/CID (0.616) despite being the same instrument as the
dominant group — this confirms a genuine fragmentation-mode effect, not just an
instrument-provenance artifact.  CID produces lower-energy, less defined fragments with more
complete bond cleavages compared to HCD's beam-type dissociation.

### Root Cause

- Training data: ~87% HCD, ~13% CID → model learns HCD peak patterns almost exclusively.
- CID peak distributions are broader, more low-m/z intense, and less reproducible than HCD.
- IT/CID completely absent from NIST23 training; all 35 test samples are from MoNA23.

### Plan

#### Step 1 — Source additional CID data

- Download GNPS public datasets filtered to `frag_mode=CID` with high-resolution instruments.
- Include MoNA spectra beyond the current MoNA23 split, specifically IT/CID entries.
- Target: 2× current CID training count minimum.

#### Step 2 — Frag-mode conditioning token

The model currently sees `inst_type=FT` for both FT/HCD and FT/CID spectra and cannot
distinguish them.  Adding `frag_mode` as a separate conditioning signal is the direct fix.

**1. Dataset** (`base_dataset.py` / `spec_mol_frag_dataset.py`):
- Load `frag_mode` column from spectrum metadata.
- Map to integer index: `HCD=0, CID=1, unknown=2`.
- Store as `spec_frag_mode` integer tensor in the batch dict.
- Fall back to `unknown=2` when `frag_mode` is missing or not in the known set.

**2. Model** (`base_model.py`):
- Add a `FragModeModel` class mirroring the existing `InstModel`.
- `nn.Embedding(num_frag_modes + 1, frag_mode_insert_size)` — `+1` for unknown token.
- Insert at `mlp` location (same as `inst_type`) so it conditions the intensity head.
- `frag_mode_insert_size: 8` is sufficient (3 categories: HCD, CID, unknown).

**3. Config**:
```yaml
frag_mode: True
frag_mode_insert_size: 8
frag_mode_insert_location: mlp
frag_modes: [HCD, CID]   # unknown handled as fallback index
```

This allows the model to shift its predicted peak distribution based on fragmentation
physics: CID produces more low-mass fragment ions and less reproducible relative intensities
compared to HCD's beam-type dissociation.

#### Step 3 — CID-specific intensity scaling

CID spectra tend to have different relative intensity distributions vs HCD.  Add a
mode-conditioned output scaling layer (a learned affine transform per frag\_mode) applied
after the intensity prediction head.

#### Step 4 — IT instrument support

IT spectra require a separate consideration: the mass accuracy is much lower (~0.3 Da vs.
~5 ppm for Orbitrap).  The `cos_sim_0.01` metric (0.01 Da tolerance) is too strict for IT.
Either:
- Use `cos_hun` (Hungarian matching) as the primary metric for IT predictions, or
- Add IT-specific peak matching tolerance as a conditioning signal.

---

## 3. Dual CE Input (ACE + NCE, No Conversion)

### Problem with Current Approach

`fill_missing_nce()` converts ACE → NCE for QTOF using
`nce = ace * charge / (prec_mz / 500)`.  The ACE↔NCE correlation across all spectra is
only **0.779** — the conversion is lossy and introduces m/z-dependent noise for QTOF.

The dataset no longer enforces mutual exclusivity (old assert removed). Whenever either
`spec_params.nce` or `spec_params.ace` is enabled, both CE columns are computed and batched as
`spec_nce` and `spec_ace`. `spec_ce` remains the primary CE tensor (`nce` when available,
otherwise `ace`) for backward compatibility.

Dual-CE model support is now implemented through the existing CE path: set `use_nce: True`
and `use_ace: True`. `CEModel` embeds NCE and ACE with independent modules and concatenates
the embeddings. `FraGNNetModel.forward()` passes `spec_ace` / `spec_ace_batch_idxs` into
`embed_ce()` whenever `use_ace=True`.

CE coverage in test set:

| Coverage | Count | % |
|---|---|---|
| Both NCE and ACE | 62,333 | 72.8% |
| Only NCE | 15,865 | 18.5% |
| Only ACE (QTOF-heavy) | 7,462 | 8.7% |

### Current Approach: Feed Both, No Conversion

Pass ACE and NCE as **two separate scalar inputs**. The model embeds each independently and
concatenates them. Missing values use the existing `-1` sentinel handled by the CE transforms
(`bin` maps it to the missing bucket; embedding-style CE inputs preserve it as the clamped
missing sentinel). This preserves the two CE scales and lets the model learn
instrument-specific reliability from data.

| Spectrum | ace | nce | What model sees |
|---|---|---|---|
| FT/HCD | NaN or real | real | `[missing_ace, nce_embed]` |
| QTOF/CID ACE-only | real | NaN | `[ace_embed, missing_nce]` |
| Both available | real | real | `[ace_embed, nce_embed]` |

### Implementation

**1. Dataset** (`base_dataset.py`): ✅ **Done**
- Old `assert not (ace and nce)` guard removed.
- Both CE columns computed when either is requested (vectorised fill via `_fill_ce_scalar_vectorized`).
- `ce_key="nce"` when both are available; model still receives `spec_ce` for compatibility.
- Batches always include `spec_nce` / `spec_nce_batch_idxs` and `spec_ace` / `spec_ace_batch_idxs`
  when CE is enabled.
- `-1` sentinel for missing values handled in `fill_missing_ace` / `fill_missing_nce`.

**2. Model** (`base_model.py`, `fragnnet_model.py`): ✅ **Done**
- `CEModel._setup_ce()` builds independent NCE and ACE embedders when `use_nce` / `use_ace`
  are enabled.
- `CEModel.embed_ce()` concatenates active CE embeddings, so `ce_insert_size: 32` gives a
  64-dim CE input when both NCE and ACE are active.
- `FraGNNetModel.forward()` reads `spec_ace` and `spec_ace_batch_idxs` from the batch when
  `use_ace=True` and passes them into both the CE embedding and CE FiLM scaler paths.

**3. Config**:
```yaml
# dataset: enabling nce is enough to compute both columns; ace can be enabled too if desired
spec_params:
  nce: True
  ace: True

# model: embed both independently and concatenate
use_nce: True
use_ace: True
ce_insert_size: 32      # per-CE; concat output = 64
ce_insert_location: mlp
ce_insert_type: embed_clamped
```

### Why No Conversion Is Better

- ACE (eV) and NCE (%) encode physically different quantities — absolute beam energy vs
  mass-normalised energy.  They carry partially independent information (r=0.779).
- QTOF natively reports ACE; converting to NCE via `prec_mz` adds m/z-dependent noise.
- FT natively reports NCE; it already has reliable NCE — the ACE path is absent or noisy.
- The model implicitly learns "trust ACE for QTOF, trust NCE for FT" from training data
  without any hand-coded conversion coefficients or per-instrument scaling factors.

---

## 3a. CE as Temperature — FiLM Logit Scaler

### Motivation

The existing `ce_insert_location: mlp` approach embeds CE as an *additive input feature*
concatenated to fragment embeddings before the MLP head.  The network must then implicitly
learn to use this feature to produce different output distributions.  This is indirect:
CE physically controls which fragments appear and at what intensity, which is fundamentally
a *multiplicative* effect on the output distribution — directly analogous to temperature
in a Boltzmann distribution.

Concretely:

- **Low CE** → little energy deposited → molecule barely fragments → distribution
  concentrated on a few large fragments (sharp).
- **High CE** → many bonds broken → different H-shift variants activated (e.g. H-loss,
  water-loss fragments) → different subset of fragments dominate.

This is not just a uniform sharpening/flattening (pure temperature) — some fragment
variants *appear* at high CE that were absent at low CE, while others *disappear*.

### Design: `CEScaler` (FiLM on H-shift logits) [2, 3]

**Class**: `CEScaler` in `src/fragnnet/model/base_model.py`

The model output before softmax has shape `(num_nodes, mlp_output_dim)` where
`mlp_output_dim = 2 * num_hs + 1`.  Each column corresponds to a proton-transfer
variant of each fragment node:

```
columns: [Δ0, Δ-1, Δ+1, Δ-2, Δ+2, ..., Δ-num_hs, Δ+num_hs]
```

`CEScaler` predicts per-H-shift `(log_scale, bias)` from normalized CE via a 2-layer MLP
and applies:

```
logits_out = logits * exp(log_scale(CE)) + bias(CE)
```

before the `scatter_logsoftmax`.  This lets the model learn physics like:
- At high CE: upweight Δ-1, Δ-2 columns (H-loss / water-loss fragments become prominent)
- At low CE: upweight Δ0 column (intact-formula fragments dominate)

**Implementation details:**

| Aspect | Detail |
|---|---|
| MLP architecture | `Linear(1, 64) → SiLU → Linear(64, output_dim * 2)` |
| Init | Last-layer weights and biases set to zero → identity at training start |
| Missing CE (-1) | Masked to 0 before normalization → identity scaling preserved |
| Multi-CE per sample (ramped/stepped) | Scatter mean over CE dim before expanding to nodes |
| Applied | After `frag_mode_scaler`, before `frag_joint_logits.flatten()` |

**Two complementary CE conditioning roles:**

| Mechanism | Controls | Config |
|---|---|---|
| `ce_insert_location: mlp` | Which *fragment nodes* the MLP selects (representation level) | `ce_insert_location: mlp`, `ce_insert_type: lin` |
| `ce_scale: True` (new) | Which *H-shift columns* are preferred at a given CE (output level) | `ce_scale: True`, `ce_scaler_hidden_dim: 64` |

Both should be active together for full CE conditioning.

### Config

```yaml
# CE as input feature (existing — representation conditioning)
ce_insert_location: mlp
ce_insert_type: lin
ce_insert_size: 32

# CE as temperature (new — output distribution conditioning)
ce_scale: true
ce_scaler_hidden_dim: 64
```

### Why FiLM, not pure temperature? [3]

A scalar temperature `logits / τ(CE)` would uniformly sharpen or flatten all H-shift
columns equally.  FiLM [2] allows *non-uniform* modulation: H-loss columns can be amplified
while the Δ0 column is suppressed, matching the actual physics of CE-dependent
fragment appearance and disappearance.  The Distill article [3] further shows that
multiplicative feature-wise conditioning is consistently more expressive than additive
concatenation — motivating `ce_scale` over relying solely on `ce_insert_location`.

### Key files

- `src/fragnnet/model/base_model.py` — `CEScaler` class
- `src/fragnnet/model/fragnnet_model.py` — `__init__` wiring (`ce_scale`, `ce_scaler_hidden_dim`), `forward` application
- `src/fragnnet/pl_model/fragnnet_pl.py` — hparam passthrough
- `config/template.yml` — default `ce_scale: false`, `ce_scaler_hidden_dim: 64`
- `tests/test_ce_scaler.py` — unit tests (identity init, FiLM scaling, missing CE, multi-CE aggregation)

---

## 3b. Adduct-Type FiLM and Categorical Stratum FiLM

### Motivation

The implemented `CEScaler` handles continuous collision-energy effects: how spectrum
severity changes as CE changes.  It does **not** directly model categorical acquisition
and ionization differences:

- `inst_type`: FT / QTOF / IT
- `frag_mode`: HCD / CID / unknown
- `prec_type`: `[M+H]+`, `[M+Na]+`, `[M+Cl]-`, `[M-H]-`, etc.

These strata have systematic output-distribution differences that are not just CE effects.
For example, QTOF/CID and rare adducts can have different fragment-intensity priors even
at similar CE.  Current embedding concatenation gives the MLP a token, but the same output
head must still serve all strata.  A small categorical FiLM module would let each stratum
apply output-level corrections while preserving the shared FraGNNet backbone.

### Adduct-Type FiLM: First Experiment

Start with adduct-only FiLM because it is cheap and directly targets known weak groups
(`[M+Na]+`, `[M+Cl]-`, and underperforming negative mode):

```
Adduct FiLM:    prec_type token -> H-shift logit scale/bias
```

Apply after `formula_module` and after existing `frag_mode_scaler` / `ce_scaler`, before
`scatter_logsoftmax`:

```python
gamma, beta = self.adduct_scaler(prec_embed).chunk(2, dim=-1)
frag_joint_logits = frag_joint_logits * (1 + gamma[frag_batch]) + beta[frag_batch]
```

Recommended config knobs:

```yaml
adduct_scale: true
adduct_scaler_hidden_dim: 32
```

Implementation should mirror `CEScaler` identity initialization: final projection weights
and biases start at zero, so enabling the module initially preserves the current model.

### Combined Stratum FiLM: Follow-Up

Keep this separate from CE FiLM for clean ablation:

```
CE FiLM:          continuous CE -> H-shift logit scale/bias
Adduct FiLM:      prec_type token -> H-shift logit scale/bias
Stratum FiLM:     inst/frag/adduct tokens -> H-shift logit scale/bias
```

Apply after `formula_module` and after `frag_mode_scaler` / `ce_scaler`, before
`scatter_logsoftmax`:

```python
stratum_embed = inst_embed + frag_mode_embed + prec_embed
gamma, beta = self.stratum_scaler(stratum_embed).chunk(2, dim=-1)
frag_joint_logits = frag_joint_logits * (1 + gamma[frag_batch]) + beta[frag_batch]
```

Recommended first version:

| Aspect | Detail |
|---|---|
| Inputs | `spec_inst_type`, `spec_frag_mode`, `spec_prec_type` |
| Apply location | Formula/H-shift logits, not graph embeddings |
| Output dim | `mlp_output_dim` (`2 * num_hs + 1`, doubled when CMF is active) |
| Architecture | Tiny MLP or embedding tables + linear projection |
| Init | Identity: final weights/biases zero so training starts from current model |
| Ablation | CE FiLM only vs adduct FiLM only vs stratum FiLM vs combinations |

### Why This Is Different From Existing Frag-Mode Conditioning

`frag_mode_insert_location: mlp` is representation conditioning: it gives the MLP a
categorical input feature.  Categorical FiLM is output-distribution conditioning: it
directly changes the relative preference for H-shift columns and formula logits per
stratum.  This mirrors the rationale for CE FiLM, but uses categorical acquisition
metadata instead of continuous CE.

### Evaluation

Track aggregate metrics and the strata this module is meant to help:

- `val/cos_sim_Mplus_Na`
- `val/cos_sim_Mplus_Cl`
- `val/cos_sim_Mminus_H`
- `val/cos_sim_QTOF_CID`
- `val/cos_sim_FT_CID`
- negative-ion vs positive-ion buckets

If adduct FiLM improves rare adducts while leaving `[M+H]+` flat, keep it.  If combined
stratum FiLM then further improves QTOF/CID or FT/CID without hurting dominant FT/HCD,
it is worth the extra categorical conditioning.  If either module only improves aggregate
metrics by exploiting dominant strata, prefer sampler/loss reweighting first.

---

## 4. DAG Truncation for Large / Polycyclic Molecules

### Constraint

All fixes must be **structure-only** — they must work at inference time from SMILES alone,
with no observed spectrum.  MAGMA is not usable here because it scores fragments against
the observed spectrum, which is unavailable at inference.

### Formula Recall Baseline (from `formula_ann_exp.ipynb`)

Comparison of formula-level annotation recall (AR) and precision (AP) across models,
measured against ground-truth MS/MS annotations from the NIST23 scaffold split.
**Evaluated on merged spectra (all CEs combined per molecule) — absolute values are higher
than per-CE evaluation; use for relative model comparisons only.**

| Model | True Recall (AR) | Weighted Recall (AWR) | Pred Precision (AP) | Weighted Precision (AWP) | F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| FraGNNet-D4 | 0.8088 | 0.8997 | 0.9837 | 0.9991 | 0.8877 | 0.9468 |
| GrAFF | **0.9509** | **0.9804** | 0.6106 | 0.8022 | 0.7437 | 0.8824 |
| ICEBERG | 0.6559 | 0.8094 | 0.9926 | — | 0.7899 | — |

**FraGNNet-D4 misses ~19% of ground-truth annotated fragment formulas** (AR=0.8088 vs GrAFF
AR=0.9509). GrAFF's higher recall comes at the cost of precision (AP=0.61 vs 0.98) — it
over-generates fragments.  The target for improved BFS is **AR > 0.90 while keeping AP
> 0.95**.  Formula-recall improvement is the primary quantitative goal of §4.

Weighted metrics favour FraGNNet-D4: AWP=0.9991 vs GrAFF AWP=0.8022.  The precision gap
means GrAFF's additional recalled fragments are predominantly low-intensity peaks — the
model correctly de-emphasises them by intensity weighting.

### Problem

Performance degrades with molecular size (MW > 350 Da, num\_atoms > 30) and is severe for
polycyclic classes.  Two distinct failure modes:

**A. Timeout truncation** — combinatorial explosion of bond-break paths hits the 150 s
budget.  When `force_stop=True` the DAG is pruned to `reached_depth - 1`.  The most
likely candidates based on size-adjusted residuals:

| Class | Median | Residual | Why likely truncated |
|---|---|---|---|
| Anthracenes | 0.276 | -0.439 | 3 fused aromatic rings → exponential ring-bond combos |
| Steroids | 0.595 | -0.152 | 4 fused rings, 90 unique molecules |
| Morphinans | 0.380 | -0.299 | Polycyclic alkaloid scaffold |

**B. Formula redundancy** — large molecules produce many fragments sharing the same
molecular formula but different structures; the model's formula-matching signal becomes
noisy.

### Plan

#### Step 1 — Audit `force_stopped` rates per chemical class (diagnostic, do first)

```python
dag_stats = load_dag_hdf5_stats()  # force_stopped, reached_depth, dag_num_nodes per mol
df_audit = df.merge(dag_stats, on='mol_id')
print(df_audit.groupby('klass')[
    ['force_stopped', 'dag_num_nodes', 'cos_sim_0.01']
].agg(['mean', 'median']))
```

Gate Steps 2–4 on this result.  Only invest in BFS changes if anthracenes / steroids show
high `force_stopped` rates.

#### Step 2 — Prioritized BFS (inference-compatible)

Replace uniform-bond BFS with a bond-priority queue scored purely from molecular structure.
Higher-priority bonds are broken first so the most chemically meaningful fragments survive
even when timeout is reached:

| Bond type | Priority |
|---|---|
| Heteroatom-adjacent (N–C, O–C, S–C) | Highest |
| Benzylic / allylic C–C | High |
| Labile ring bonds (glycosidic, ester, lactone) | High |
| Generic ring C–C | Low |
| Aliphatic chain C–C | Lowest |

This is purely structure-derived and works identically at training and inference time.
Implementation: `compute_frags.pyx` — replace the flat bond list with a priority-sorted
list before the BFS depth loop.

#### Step 3 — One-shot 2-bond ring cleavage pre-pass

**Root cause:** Breaking one bond in a fused ring never disconnects the molecule (the ring
closure provides an alternative path), so `connected_components` always returns `num_ccs=1`
and no new DAG node is created.  For naphthalene at depth 1: all 11 bond masks produce the
same atom set — **zero new fragments**.  Meaningful ring products only appear at depth 2+,
which is where timeout hits polycyclics.

**Fix:** Before the main BFS depth loop in `compute_frags.pyx`, run a pre-pass that
enumerates all **pairs of ring bonds** whose *simultaneous* removal disconnects the molecule
(`num_ccs == 2`), and inserts the resulting child fragments as synthetic **depth-1** nodes.

```python
# pre-pass (Python-side, runs once per molecule, before compute_ccs)
ring_bond_ids = [i for i, b in enumerate(mol.GetBonds()) if b.IsInRing()]
for i, j in combinations(ring_bond_ids, 2):
    edge_mask[i] = 0; edge_mask[j] = 0
    cc_nodes, cc_edges, num_ccs = connected_components(...)
    edge_mask[i] = 1; edge_mask[j] = 1
    if num_ccs == 2:
        inject both children into depth-1 frontier of compute_ccs
```

This directly models **retro-Diels-Alder** (RDA), fused-ring junction cleavage, and
lactone ring opening — all 2-bond events that currently require sequential depth-2+ BFS.

**Depth savings per class:**

| Molecule | Current first ring fragment depth | With one-shot |
|---|---|---|
| Naphthalene (2 fused rings) | Depth 2 | Depth 1 |
| Anthracene (3 fused rings) | Depth 2–3 (often timeout) | Depth 1 |
| Steroid A-ring RDA | Depth 4+ (timeout) | Depth 1 |
| Coumarin lactone + benzene | Depth 3–4 | Depth 1 |

**Complexity:** `O(R²)` at the root only, where R = number of ring bonds.  For a steroid
with ~20 ring bonds, that's ≤ 190 connectivity checks — negligible vs. the current BFS.

**Implementation notes:**
- `ring_bond_mask` (bool array per bond) computed in `extract_mol_info` from
  `bond.IsInRing()` and passed into `compute_dags` / `compute_ccs`.
- Injected children enter `ccs` with `depth = 1` and participate in the normal BFS from
  depth 2 onward — no changes to the BFS core logic.
- Inference-safe: ring membership is structure-only, no spectrum needed.

#### Step 4 — Ring-contraction strategy for polycyclics

For molecules with ≥ 3 fused rings, add an optional fragmentation pass that treats each
fused ring system as an atomic unit (does not break internal ring bonds at depth 1).
This dramatically reduces the BFS branching factor while still enumerating all fragments
that result from breaking bonds at ring junctions or on external chains.  Inference-safe:
ring system detection is structure-only.

#### Step 5 — Expose `force_stopped` and `dag_num_nodes` as model inputs

```python
mol_features = concat([existing_features, log(dag_num_nodes + 1), force_stopped_flag])
```

The model learns to adjust confidence when it knows the DAG is incomplete.  Both values
are computed at preprocessing time and available at inference.

#### Step 6 — Neutral loss representation for high-MW molecules

For molecules above a MW threshold, represent DAG edges as neutral losses (Δformula from
parent) rather than absolute fragment formulas.  Neutral losses are more structurally
specific than absolute formulas at high MW and reduce formula-redundancy noise.

#### Step 7 — 128-atom hard limit audit

Check what fraction of training/test molecules hit the `MAX_NUM_NODES=128` assertion in
`compute_frags.pyx`.  If non-trivial, extend the limit and reprocess.

---

## 4a. DAG Expansion Experiments — Prepass and MBFS (Negative Results)

### Experiments Run (2026-04-06/07, NIST20 inchikey split, 100 epochs)

The prepass/ruleid DAG files were regenerated with mass-based pruning after the first round
of experiments.  The updated `smpTrue` and `smpTrue_ruleid` tarballs shrank from 2.4 GB to
1.7 GB — identical to the baseline — while the MBFS variants remain larger.

| Job | DAG data | Size | Extra feature | Val cos_sim (best) | Val loss (best) | Status |
|---|---|---|---|---|---|---|
| `all` (baseline) | `smpFalse` | 1.7 GB | — | **0.6688** | **−2.416** | Completed |
| `prepass` | `smpTrue` (updated) | **1.7 GB** | — | 0.6621 | −2.373 | Completed |
| `prepass_ruleid` | `smpTrue_ruleid` (updated) | **1.7 GB** | legacy rule-id variant | 0.6623 | −2.365 | Completed |
| `mbfs` | `smpFalse_mbfs` | 2.2 GB | — | — | — | OOM Killed |
| `mbfs_prepass` | `smpTrue_mbfs` | 2.5 GB | — | — | — | OOM Killed |

Val loss is negative log-likelihood — more negative is better. Baseline wins on both metrics.

### DAG Coverage Stats (mol_stats_df, tol=0.01 Da, h1)

| Dataset | DAG nodes (mean/median) | recall_h1 | wrecall_h1 | prec_h1 | noise_ratio |
|---|---|---|---|---|---|
| smpFalse (baseline) | 764 / 280 | 0.341 | 0.456 | 0.071 | 0.929 |
| smpTrue (prepass) | 769 / 282 | 0.345 | 0.460 | 0.071 | 0.929 |
| smpTrue_ruleid | 769 / 282 | 0.325 | 0.405 | 0.100 | 0.900 |
| smpFalse_mbfs | 1816 / 402 | 0.402 | 0.533 | 0.062 | 0.938 |
| smpTrue_mbfs | 2197 / 421 | 0.419 | 0.547 | 0.062 | 0.938 |

`prec_h1` = fraction of DAG nodes matching at least one observed peak.
`noise_ratio` = 1 − prec_h1 = fraction of DAG nodes that match no observed peak.

### Key Findings

**1. Prepass DAG coverage stats are nearly identical to baseline.**
After pruning, smpTrue has almost the same node count (769 vs 764), the same precision
(0.071), the same noise ratio (0.929), and marginally better recall (+0.4%).  The coverage
stats give no reason to expect degradation — yet val cos_sim drops by 0.007.  This rules
out quantity-of-noise as the cause.  The degradation must come from the *character* of the
prepass fragments (graph topology, connectivity pattern) rather than their count.

**2. smpTrue_ruleid has anomalous stats despite the same DAG structure.**
Same dag_num_nodes/edges as smpTrue, but lower recall (0.325 vs 0.345) and higher precision
(0.100 vs 0.071).  This is paradoxical and likely indicates a labeling/matching artifact in
how the ruleid dataset's stats were computed — possibly a different matching criterion or a
data pipeline bug.  The val cos_sim is essentially the same as prepass (0.6623 vs 0.6621),
confirming the ruleid feature adds nothing.

**3. MBFS has genuine coverage gains but at the cost of 2.4× more nodes and lower precision.**
mbfs recall = 0.402 (+18% over baseline), wrecall = 0.533 (+17%), but prec drops to 0.062
(−13%) and median nodes jump from 280 → 402.  The coverage gain is real but diluted by
proportionally more noise.  MBFS still OOM at 2.2 GB — the cluster memory budget sits just
above the 1.7 GB baseline threshold.

### Possible Causes for Persistent Prepass Degradation

The stats confirm prepass and baseline DAGs have nearly identical coverage properties
(same node count, same noise ratio, marginally better recall).  The degradation is therefore
not about fragment *quantity* but fragment *quality* — specifically how the model embeds and
ranks SMARTS-generated fragments vs BFS-generated ones.

1. **Rearrangement fragments are out-of-distribution for the GNN.**
   SMARTS-generated fragments result from bond rearrangements (1,3-eliminations, retro-ester,
   CO₂ loss, etc.).  They can share the same atom set as a BFS fragment but have different
   bond connectivity (e.g., a ring-opened form vs the closed form).  The GNN has never seen
   these subgraph patterns during training on BFS-only DAGs, so their embeddings are
   unreliable — they look chemically plausible but score incorrectly.

2. **DAG message-passing corrupted by atypical depth-1 nodes.**
   Prepass fragments are injected at depth 1 as synthetic children of the precursor.  GNN
   message-passing aggregates over all children before computing the precursor embedding.
   If some depth-1 nodes have unusual subgraph structure, the aggregated precursor
   representation is perturbed — affecting every fragment score in the DAG, not just the
   prepass ones.

3. **The model has no way to distinguish BFS from SMARTS fragments.**
   Both fragment types look like ordinary molecular graphs to the GNN.  The ruleid experiment
   confirms that providing an explicit label doesn't help (arbitrary index, not learnable).
   The model cannot learn "SMARTS fragments need different scoring" without a structural
   inductive bias — it would need a better embedding strategy (see §4b below).

4. **Gradient interference from label ambiguity.**
   When a prepass fragment has the same nominal mass as a BFS fragment (mass collision), both
   compete for the same observed peak.  Across training examples, this creates conflicting
   gradients: push the prepass fragment up for molecule A (where it's the true fragment),
   push it down for molecule B (where BFS is correct).  Net effect: both fragment types are
   scored slightly worse.

### Correct Direction: DAG Pruning, Not Expansion

**Fragment coverage is not the bottleneck** — the baseline BFS already generates enough
candidate fragments.  The bottleneck is precision: wrong candidates (whether from prepass
or MBFS) dilute the signal regardless of their count.

Instead of adding more fragments, the correct direction is to reduce DAG noise by pruning:

1. **Mass filter on baseline BFS DAGs** — drop BFS fragments whose neutral mass (±
   H-transfer range) has no matching peak in the spectrum's m/z window.  This shrinks
   baseline DAGs without changing their structural character.
2. **Depth/cut limit** — reducing max BFS depth from d3→d2 would shrink DAGs significantly;
   measure coverage loss on training set before applying.
3. **Top-k by structural simplicity** — keep only the k least-complex fragments at each
   depth level.

---

## 4b. Better Fragment Embedding for Rearrangement Products

### Problem

The prepass stats show rearrangement fragments have identical coverage properties to BFS
fragments (same noise ratio, slightly better recall), yet the model scores them worse.
The root cause is that the GNN embeds all fragments from their molecular graph alone —
it cannot tell whether a fragment was produced by simple bond cleavage (BFS) or a bond
rearrangement (SMARTS prepass), and it has no inductive bias to score them differently.

### Option 1 — Fragment origin as a learned node embedding (low effort, try first)

Add a binary or categorical `frag_type` node feature to every atom in each fragment:
`0 = BFS-generated`, `1 = SMARTS prepass`.  Unlike the integer `rule_id` (which was
arbitrary and unlearnable), a binary flag provides a clean inductive bias: the model can
learn a scalar adjustment to fragment scores conditioned on whether they came from a
rearrangement.

```python
# in dag feature extraction
frag_type = torch.zeros(num_atoms, dtype=torch.long)  # 0 = BFS
if is_prepass_fragment:
    frag_type[:] = 1  # 1 = SMARTS rearrangement
node_feats = torch.cat([existing_feats, F.one_hot(frag_type, 2).float()], dim=-1)
```

**Why this is better than ruleid:** the model only needs to learn one scalar per type
(two types), not a separate scalar per rule (N rules).  Much lower-variance gradient.

### Option 2 — Bond-breaking type as edge feature (medium effort)

Encode *how* each bond in the parent molecule was broken to produce this fragment:
`0 = not broken`, `1 = simple BFS cut`, `2 = SMARTS rearrangement cut`.  This is richer
than a per-fragment flag because the model can attend to which specific bonds distinguish
BFS from rearrangement topology.

Requires passing the bond-breaking record from the fragmentation code into the DAG
feature extraction pipeline.

### Option 3 — Separate GNN tower with late fusion (high effort)

Run two separate GNN encoders — one for BFS fragments, one for prepass fragments — and
fuse their outputs at the DAG aggregation step.  The BFS tower is trained on the large,
clean BFS signal; the prepass tower learns from the smaller rearrangement signal without
the two interfering.

This is architecturally sound but expensive: doubles fragment encoder parameters and
requires routing each DAG node to the correct tower.  Only worth pursuing if Options 1
and 2 fail.

### Option 4 — Contrastive pre-training of fragment encoder (high effort)

Pre-train the fragment GNN with a contrastive objective: BFS fragments of the same
molecule should be closer to each other than to prepass fragments of unrelated molecules.
Provides a structural inductive bias before any spectrum supervision.

### Recommended order

1. Try **Option 1** (frag_type binary flag) — one line of feature code, reuse existing
   configs, cheapest experiment.  If prepass val cos_sim recovers to ≥ baseline, done.
2. If Option 1 fails, try **Option 2** (bond-breaking edge feature) — requires fragmentation
   code changes but provides richer signal.
3. Options 3 and 4 only if structural embedding is confirmed as the bottleneck and
   Options 1/2 provide no gain.

---

## 5. Chemical Class Fragmentation Failures

### Wrecall Metrics Used in This Section

Two complementary metrics from `predictions_frag_explainability.ipynb`:

- **`wrecall_0.01_h1`** (primary for diagnosis): intensity fraction covered within ±0.01 Da with max 1 H-shift.  H1 is strict and therefore discriminating — used for chemical-class comparisons in the plan tables.
- **`wrecall_10ppm_h4`** (relaxed): same but 10 ppm and up to 4 H-shifts.  Higher overall (median 0.932 at mol-level vs 0.722 for h1) but **not recommended as the primary metric**: allowing h4 *hurts* worst-case performance (p10) for lipids (−0.255) and organoheterocyclics (−0.184), indicating over-matching of spurious fragment formulas.  **Cap H-shifts at h2 or h3 for production inference.**

### Superclass-Level Wrecall Analysis (from `predictions_frag_explainability.ipynb`)

**Mol-level `wrecall_10ppm_h4` by ClassyFire superclass** (1,486 molecules — scaffold split,
**merged spectra**; one combined spectrum per molecule across all CEs):

| Superclass | Median wrecall_h4 | N Molecules | Notes |
|---|---|---|---|
| Organic polymers | 0.808 | 3 | Atom-count limit + OOD; deprioritize (see below) |
| Organosulfur compounds | 0.887 | 31 | Rearrangement neutral losses (SO₂, H₂S…); neutral loss pre-pass needed (see below) |
| Organophosphorus compounds | 0.931 | 17 | At overall median |
| Hydrocarbons | 0.942 | 23 | Few heteroatom anchors |
| Organic oxygen compounds | 0.955 | 265 | McLafferty (ketones/esters) + H₂O loss (alcohols) missing; worst tail of all superclasses (33.5% below 0.9) |
| Lipids | 0.968 | 3,957 | High median but high variance (see below) |
| Organoheterocyclic | 0.974 | 3,038 | — |
| Benzenoids | 0.989 | 677 | — |
| Alkaloids | 0.997 | 87 | Best overall |
| Organic acids | 0.999 | 598 | — |

**`[M+H]+` FT spectra only — fraction with `wrecall_h4 < 0.9`** (6,597 spectra,
**per-CE spectra** from scaffold split test set):

| Superclass | Median wrecall | Frac < 0.9 | Frac < 0.5 |
|---|---|---|---|
| Organic oxygen compounds | 0.875 | 33.5% | 11.7% |
| Lipids | 0.902 | 32.6% | 10.9% |
| Organoheterocyclic | 0.925 | 32.4% | 8.6% |
| Organosulfur | 0.943 | 24.1% | 10.3% |
| Alkaloids | — | 19.0% | 5.1% |
| Benzenoids | — | 15.8% | 3.2% |

**Key finding:** Even for the "easy" Lipid superclass, ~11% of `[M+H]+` FT spectra have <50% of intensity covered by the DAG — these are the molecules most likely to benefit from one-shot 2-bond ring cleavage or explicit fragmentation rules.

### H-Shift Sensitivity — Do Not Use h4

| H-shifts | Median wrecall (mol) | p10 wrecall (Lipids) | p10 wrecall (Heterocyclics) |
|---|---|---|---|
| h1 | 0.722 | — | — |
| h2 | 0.844 | — | — |
| h3 | 0.901 | — | — |
| h4 | 0.932 | **−0.255 vs h1** | **−0.184 vs h1** |

Allowing h4 inflates the median but degrades the worst 10% of molecules.  This indicates spurious formula matches at the 4-H-shift level.  Use h2 as the default, h3 as maximum.

### New Finding: DAG Coverage Gaps for Fused Rings and Steroids

`predictions_frag_explainability.ipynb` using the `wrecall_0.01_h1` metric (fraction of spectral
intensity explainable by DAG formulas, strict ±0.01 Da / 1 H-shift) reveals severe coverage gaps
for specific chemical classes. **Evaluated on merged spectra — wrecall values are higher than
per-CE equivalents.**

**High Explainability (DAG works well):**
- Nitrogen-containing heterocycles: `azoles` (0.698), `pyrrolopyridines` (0.659), `diazines` (0.612)
- Lipids: `prenol lipids` (0.571), `fatty acyls` (0.568)

**Low Explainability (DAG is missing crucial fragments):**
- Fused Ring Systems: `coumarins and derivatives` (0.358), `diazanaphthalenes` (0.376), `indoles and derivatives` (0.421)
- Steroids: `steroids and steroid derivatives` (0.416) — highly significant due to large class size (n=112 in top 20).

**Conclusion:** The DAG generation algorithm struggles heavily with **polycyclic/fused-ring structures**. Breaking a single bond in a fused ring doesn't split the molecule; it requires multiple coordinated bond breaks or complex rearrangements (e.g., retro-Diels-Alder, water losses, multi-bond ring cleavages) that simple bond-breaking rules fail to capture within the depth/timeout limits.

### Revised Understanding of RDA and Rearrangements

While some simple RDA products (e.g., in flavonoids) might be reachable within `max_depth=4` by breaking two bonds sequentially, the low `wrecall` for steroids and coumarins proves that **many complex rearrangement products are indeed missing from the DAG**. 

For classes where the masses *are* present (high `wrecall` but low `cos_sim`), the problem is **intensity misprediction**: the model hasn't learned to assign high intensity to the specific fragments corresponding to preferred pathways.

Consequence: We **do** need explicit fragmentation rules for ring-opening events, complex rearrangements, or multi-bond cleavages characteristic of steroids and fused heterocycles to improve mass coverage. For other classes, surfacing these fragments earlier in the BFS helps them receive better graph attention.

### Diagnosis Decision Tree

Run the `force_stopped` audit (§4 Step 1) first, then apply:

| `force_stopped` rate | `wrecall` | Cos\_sim | Chemistry | Root cause | Fix |
|---|---|---|---|---|---|
| High (>30%) | Low | Low | Any | DAG truncation | Prioritized BFS (§4 Step 2), one-shot ring cleavage (§4 Step 3), ring-contraction (§4 Step 4) |
| Low | Low | Low | Fused rings, steroids | Missing 2-bond ring cleavages (RDA etc.) | One-shot pre-pass (§4 Step 3) + explicit SMARTS rules |
| Low | High | Low | RDA / ring-strain | Intensity misprediction | Focal loss (§1 Step 3), bond-priority BFS |
| Low | High | Low | No special chemistry | Data imbalance | Focal loss + more training data |

### Classes and Their Expected Root Cause

| Class | Median | Residual | Expected cause |
|---|---|---|---|
| Anthracenes | 0.276 | -0.439 | DAG truncation (3 fused rings) |
| Morphinans | 0.380 | -0.299 | DAG truncation (polycyclic alkaloid) |
| Furans | 0.383 | -0.313 | Intensity misprediction — ring-opening bonds not prioritised |
| Lactones | 0.416 | -0.281 | Intensity misprediction — ring-strain cleavage not prioritised |
| 2-arylbenzofuran flavonoids | 0.360 | -0.318 | Intensity misprediction — benzofuran + RDA pattern |
| Benzopyrans | 0.511 | -0.221 | Intensity misprediction — RDA masses present, weights wrong |
| Steroids | 0.595 | -0.152 | Missing complex pathways (low `wrecall`) + DAG truncation |
| Coumarins | 0.358 | — | Missing complex pathways (low `wrecall`) |
| Indoles | 0.421 | — | Missing complex pathways (low `wrecall`) |
| Prenol lipids | 0.634 | -0.127 | Intensity misprediction — isoprene junctions not prioritised |
| Lignans / neolignans | 0.537 | — | Intensity misprediction — benzyl-ether cleavage not prioritised |
| **Stilbenes** | **0.619** | — | Intensity misprediction — trans/cis stilbene retro-Diels-Alder |
| **Hydrocarbon derivatives** | **0.428** | — | DAG priority failure — few heteroatom anchors, BFS gains little structural signal |

### Bond-Priority BFS and Explicit Rules as the Unified Fix

The prioritized BFS (§4 Step 2) addresses both truncation and intensity misprediction:
- Truncation: chemically meaningful fragments are enumerated before timeout
- Intensity: BFS order influences graph depth embedding — fragments reached earlier
  get shallower depth features, and training will associate those features with higher
  predicted intensity if the real spectra confirm it

For classes with low `wrecall` (steroids, coumarins, indoles), the **one-shot 2-bond ring
cleavage pre-pass (§4 Step 3)** is the primary fix — it injects RDA and fused-ring cleavage
products at depth 1 without requiring any SMARTS patterns.  For rare 3-bond cleavages not
covered by the pre-pass, explicit SMARTS rules remain the fallback.

Specific SMARTS patterns to detect and upweight (or apply explicit rules) for problem classes:

| Class | High-priority bond pattern / Explicit Rule |
|---|---|
| Benzopyrans / flavonoids | C-ring junction bonds: `[c,C]-[O,C]` in 6-membered O-rings |
| Lactones | Ester C–O bond: `[C](=O)-O-[C]` |
| Furans | α-bond to furan oxygen: `[c]-[O]-[c]` ring bond |
| Prenol lipids | Isoprene junction: `[C](=C)-[C]-[C]=C` |
| Lignans | Benzyl-ether: `[c]-[CH2]-O` |
| Steroids | Explicit rules for retro-Diels-Alder, water losses, and multi-bond ring cleavages |
| Coumarins / Indoles | Explicit rules for fused-ring opening events |

### Rearrangement Neutral Loss Pre-Pass (General)

The neutral loss pre-pass is not sulfur-specific — it applies to **any fragmentation requiring
new bond formation**, which BFS can never produce regardless of depth.  The dividing line is
mechanical, not elemental:

| Mechanism | New bond formed? | BFS reachable? | Fix |
|---|---|---|---|
| Retro-Diels-Alder | No (2 bond breaks only) | Yes, depth 2 | One-shot ring pre-pass |
| CO loss via C–CO bond break | No | Yes, depth 1–2 | BFS handles |
| Simple NH₃ loss (C–N break only) | No | Yes, depth 1 | BFS handles |
| **McLafferty** (γ-H transfer + β-cleavage) | Yes — new C=C in enol | **Never** | Neutral loss pre-pass (variable Δm) |
| **H₂O loss** (1,2-elimination from alcohol) | Yes — new C=C | **Never** | Neutral loss pre-pass (−18 Da, fixed) |
| **SO₂ loss** (sulfonamides/sulfones) | Yes — new C–N connectivity | **Never** | Neutral loss pre-pass (−64 Da, fixed) |
| **SO₃ loss** (sulfates) | Yes | **Never** | Neutral loss pre-pass (−80 Da, fixed) |

#### PPGB_MS2 reaction coverage analysis

The 28 fragmentation reactions used by PPGB_MS2 [1] (Supplementary Figure 1, `smiles_fragmentation.py`)
have been systematically mapped to what our BFS + H-transfer model can and cannot produce.
Key insight: BFS generates atom-set fragments; bond orders within fragments are not tracked.
Many reactions that "form new pi bonds" are therefore covered **at formula level** by the
existing ±`max_h_transfer` H flexibility — the model just needs to assign the right intensity.

| SMARTS var | Figure # | Mechanism | Coverage |
|---|---|---|---|
| `crf1_0` | R1 | HX 1,2-elimination (break B–X, A=B forms) | BFS + H-transfer=1 ✓ |
| `crf1_1` | R2 | Vinyl HX elimination → triple bond | BFS + H-transfer=1 ✓ |
| `crf2_0` | R3 | Alpha-cleavage (H on C: break A–B, B=C) | BFS + H-transfer=1 ✓ |
| `crf2_1` | R4 | Alpha-cleavage → triple bond | BFS + H-transfer=1 ✓ |
| `crf3_0` | R5 | 6-ring → diene + alkene (2 bonds break) | 2-bond ring pre-pass ✓ |
| `crf3_1` | R6 | 6-ring → diene + alkyne | 2-bond ring pre-pass + H-transfer ✓ |
| `crf4_0` | R7 | McLafferty: break β–γ bond, H from γ | BFS + H-transfer=1 ✓ (formula-level) |
| `crf5_0` | R8 | H2 loss + 1 bond break (both frags lose 1H) | BFS + H-transfer=2 ✓ (each fragment independently at −2H from natural) |
| `crf6_0` | R9 | 8-atom polyene retro-rearrangement | ❌ Too complex; deprioritize |
| `crf7_0` | R10 | 6-ring → **3 fragments** (3 bonds break) | Pre-pass (depth-1) + BFS (depth-2) ✓ |
| `crf7_1` | R11 | 6-ring → 3 frags, 1 triple bond | Same + H-transfer ✓ |
| `crf7_2` | R12 | 6-ring → 3 frags, 2 triple bonds | Same ✓ |
| `crf7_3` | R13 | 6-ring → 3 frags, 3 triple bonds | Same ✓ |
| `crf8_0` | R14 | Allylic H-shift (isomerization, 1 product) | ❌ Not fragmentation; skip |
| `crf8_1` | R15 | Bis-allylic H-shift | ❌ Same |
| `crf8_2` | R16 | Extended allylic H-shift | ❌ Same |
| `crf9_0` | R17 | CO loss from ring ketone (2 ring bonds) | 2-bond ring pre-pass ✓ |
| `crf11_0` | R18 | H2 loss from adjacent C–H bonds (1 product) | ❌ H2 neutral loss needed (see below) |
| `crf11_1` | R19 | H2 loss → triple bond | ❌ Same |
| `crf12_0` | R20 | Retro-ene: 2 bond breaks + new A–D bond | ❌ New bond formed; skip |
| `crf12_1` | R21 | Retro-ene → triple bond | ❌ Same |
| `crf13_0` | R22 | CO2 from –COOH (break R–C, H-shift=2) | BFS + H-transfer=2 ✓; explicit better |
| `crf13_1` | R23 | CO2 from formate ester H–C(=O)–O–R | BFS + H-transfer=1 ✓; explicit better |
| `crf13_2` | R24 | CO2 from acyclic ester + new R1–R2 bond | ❌ Acyclic: NOT covered. Cyclic (lactone): ring pre-pass ✓ |
| `cmf2_0` | R25 | Heteroatom-activated cleavage [N/O/S]=B + C | BFS + H-transfer ✓ |
| `cmf2_1` | R26 | Same → triple bond | BFS + H-transfer ✓ |
| `cmf7_0` | R27 | Simple heterolytic A–B → A⁺ + B⁻ | BFS ✓ |
| `cmf13_0` | R28 | 6-ring contraction → 5-ring + atom (2 bonds) | 2-bond ring pre-pass ✓ (formula-level) |

**Summary:** 18/28 reactions covered by current BFS + H-transfer (R8 is covered: each fragment
independently at H-transfer=−2).  The 2-bond ring pre-pass adds coverage for R5, R6, R10–R13,
R17, R28 (6 additional).  One reaction class requires a new neutral loss entry (R18–19: pure
H2 loss with no bond break).  R9, R14–16, R20–21, R24-acyclic are deprioritized (too complex
or not fragmentation events).

#### Phase 1 — Fixed-Δm losses (low effort, implement first)

These have a known constant Δm detectable by a single SMARTS pattern:

| Neutral loss | Δm (Da) | Functional group | SMARTS | Affected superclass | PPGB/smarts.md ref |
|---|---|---|---|---|---|
| H₂O | −18.011 | alcohols, carboxylic acids | `[OH]` | Org. oxygen, lipids | R1/R3; `nl_h2o_alc`, `nl_h2o_1` |
| CO₂ | −43.990 | carboxylic acids, formate esters | `[C](=O)[OH]` | Org. oxygen, lipids | R22, R23; `nl_co2` |
| NH₃ | −17.027 | primary/secondary amines, amides | `[NX3;H1,H2]` | Alkaloids, amino acids | `nl_nh3_1`, `nl_nh3_2` |
| H₂ | −2.016 | vicinal C–H bonds (ring dehydrogenation) | `[#6;H1,H2][#6;H1,H2]` | Polycyclics, aromatics | R18, R19 (`crf11_0/1`) |
| SO₂ | −63.962 | sulfonamides, sulfones | `[S](=O)(=O)` | Organosulfur | — |
| SO₃ | −79.957 | sulfates, sulfonates | `[S](=O)(=O)[OH,O-]` | Organosulfur | — |
| CH₃SH | −48.003 | thioethers | `[S][CH3]` | Organosulfur | — |
| S | −31.972 | thioketones | `[S;!$(S=O)]` | Organosulfur | — |
| CS₂ | −75.944 | dithiocarbamates | `[S]C(=S)` | Organosulfur | — |

H₂O loss is expected to recover a substantial fraction of the organic oxygen compounds
tail (33.5% below wrecall 0.9) since alcohols and carboxylic acids are the dominant
subclasses and elimination products are high-intensity peaks.

**CO₂ loss** (new): while BFS + H-transfer=2 can formally reach the CO₂ fragment formula
from a –COOH group, explicit depth-1 injection is more reliable — it ensures the model
sees this high-intensity product early in training rather than buried in the H-transfer
lattice at depth 2+.  SMARTS `[C](=O)[OH]` fires on all carboxylic acids and matches
`crf13_0` (R22) and `crf13_1` (R23) from [1].

**H₂ loss** (new): only R18/R19 (`crf11_0/1`) require the neutral loss pre-pass.  These
produce a **single product** (same atom set as precursor, −2H) with no bond break, so BFS
never generates the required child node (BFS requires num_ccs=2 after bond removal).  H-transfer
also cannot cover this: H-transfer shifts H *between* two charged fragment nodes, but here
there is only one node (no split).  R8 (`crf5_0`) — bond break + both fragments each at −2H
from natural — is correctly covered by H-transfer=2 applied independently to each fragment.
Implementation: inject `parent_formula − H₂` as a depth-1 DAG node with the **same atom
mask as the root** (unusual — this is a formula-only node, not a structural subset).

Implementation: inject `parent_formula − Δm_formula` as a depth-1 DAG node when the
SMARTS matches.  No changes to BFS core.  Complexity O(P) per molecule (~9 patterns).

```python
FIXED_NEUTRAL_LOSSES = [
    ("[OH]",                   18.011),  # H2O  — alcohols, acids
    ("[C](=O)[OH]",            43.990),  # CO2  — carboxylic acids / formate esters
    ("[NX3;H1,H2]",            17.027),  # NH3  — primary/secondary amines, amides
    ("[#6;H1,H2][#6;H1,H2]",   2.016),  # H2   — vicinal CH (ring dehydrogenation, R18/R19 only)
    ("[S](=O)(=O)",            63.962),  # SO2  — sulfonamides, sulfones
    ("[S](=O)(=O)[OH,O-]",     79.957),  # SO3  — sulfates
    ("[S][CH3]",               48.003),  # CH3SH — thioethers
    ("[S;!$(S=O)]",            31.972),  # S    — thioketones
    ("[S]C(=S)",               75.944),  # CS2  — dithiocarbamates
]
for smarts, delta_mass in FIXED_NEUTRAL_LOSSES:
    if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
        inject_neutral_loss_fragment(parent_formula, delta_mass, depth=1)
```

#### Phase 2 — McLafferty (variable Δm, higher effort)

McLafferty is the **most common rearrangement in EI/CID MS/MS** for carbonyl-containing
molecules (ketones, aldehydes, esters, carboxylic acids).  It affects organic oxygen
compounds and lipids — the two superclasses with the highest wrecall tail.

**Mechanism:** γ-H transfers to carbonyl O; β-C–C bond breaks simultaneously, producing
a new C=C in the alkene neutral and a new C=O in the enol fragment.  New bond formation
means BFS can never produce the enol product.

**Variable Δm problem:** The Δm equals the mass of the departing alkene (the β–γ alkyl
chain), which differs per molecule.  Cannot use a single fixed Δm.

**Implementation approach:**
1. Detect γ-keto SMARTS: `[CH2,CH3][CH2][C](=O)` in the molecule
2. For each match, identify the γ-carbon atoms and compute the formula of the γ-alkyl
   fragment that departs as alkene
3. Inject `parent_formula − alkene_formula` as a depth-1 node

Effort: medium (requires formula subtraction from matched substructure atoms, not just
fixed Δm).  Implement after Phase 1 is validated on organosulfur and organic oxygen data.

### Organic Polymers: Deprioritize, Audit Atom Limit

**Root cause:** Polymers are out-of-distribution for the BFS algorithm on two levels:
1. Very large MW / atom count likely exceeds `MAX_NUM_NODES=128` in `compute_frags.pyx`,
   silently truncating or skipping the DAG entirely.
2. Repeating units produce many fragments with identical or near-identical formulas at
   multiple depths — the DAG becomes enormous and hits the 150 s timeout.

**Action:** Deprioritize for now — n=3 in the test set is too small to draw conclusions.
Run the §4 Step 7 audit first:

```python
# Check how many training/test molecules exceed 128 atoms
n_exceed = df[df['num_atoms'] > 128].shape[0]
print(f"{n_exceed} molecules exceed MAX_NUM_NODES=128 ({n_exceed/len(df)*100:.1f}%)")
```

If the fraction is non-trivial (> 0.5%), raise the limit and reprocess.  If polymers are
the only class affected and represent < 0.1% of data, treat as out-of-domain and exclude
from wrecall benchmarks rather than investing engineering effort.

---

## 6. MoNA Domain Gap

MoNA23 performance (median 0.559) vs NIST23 (median ~0.782) is likely inflated by the 35
IT spectra (all from MoNA23, median 0.000).  Run the FT-only check before any remediation:

```python
mona_ft = df[(df['dset'] == 'mona23') & (df['inst_type'] == 'FT')]
print(mona_ft['cos_sim_0.01'].median())  # expected to be closer to NIST23 FT (~0.78)
```

- If gap is small (< 0.05): MoNA issue is entirely the IT spectra — no action needed.
- If gap remains: investigate peak normalization and intensity preprocessing differences
  between MoNA and NIST curation pipelines.

## 7. Pending Diagnostics (Run Before Any Code Changes)

These are quick queries that gate decisions in §1–§6:

| Diagnostic | Code | Gates |
|---|---|---|
| `force_stopped` audit | `df_audit.groupby('klass')[['force_stopped','dag_num_nodes','cos_sim_0.01']].agg(...)` | §4 Steps 2–3, §5 root cause |
| Formula recall (AR) per class | Run `test_fraggnn_formula.py` loop over all groups, group AR by `klass` | §4 target (AR>0.90), §5 root cause |
| `wrecall_0.01_h1` per class | `frag_cf.groupby('klass')['wrecall_0.01_h1'].median()` | §5 explicit rules for fused rings |
| MoNA FT-only check | `df[(df.dset=='mona23')&(df.inst_type=='FT')].cos_sim_0.01.median()` | §6 action |
| ACE→NCE residual per instrument | `both.groupby('inst_type').apply(lambda g: (g.nce - g.nce_converted).abs().median())` | §3 Step 1 |
| Current sampler weighting | ✅ **Done** — `simple_group_sampler` with `type=group`, `avg_per_group=5`, weights `1/spec_per_group`. No inst_type/frag_mode weighting. | §1 Step 1 |
| OOS calibration per stratum | `df.groupby(['inst_type','frag_mode'])['pred_oos_prob'].describe()` | §8 confidence filter |
| wrecall superclass tail | `df.groupby('superclass')['wrecall_h4'].quantile([0.1,0.5]).unstack()` — flag superclasses with p10 < 0.5 | §5 root cause for lipid/oxygen tail |
| CE-performance curve (FT) | `df[df.inst=='FT'].groupby(pd.cut(df.nce,[0,20,40,65,120,200]))['cos_sim_0.01'].median()` | Validates sampler CE bin targets |
| CE-performance curve (QTOF) | same with ace bins [0,10,20,30,50] | §1 QTOF upweighting priority |

### CE Performance Reference (from `predictions_test_analysis.ipynb`)

**FT (NCE %) — median cos_sim_0.01 by CE bin:**

| NCE range | Median cos_sim | N spectra |
|---|---|---|
| 0–20% | **0.934** | 112,779 |
| 20–30% | 0.889 | 66,051 |
| 30–40% | 0.714 | 124,576 |
| 40–60% | 0.810 | 119,781 |
| 60–80% | 0.709 | 68,060 |
| 80–120% | 0.680 | 108,569 |
| 120–200% | 0.642 | 83,441 |

**QTOF (ACE eV) — median cos_sim_0.01 by CE bin:**

| ACE range | Median cos_sim | N spectra |
|---|---|---|
| 0–10 eV | **0.864** | 13,584 |
| 10–20 eV | 0.619 | 8,749 |
| 20–30 eV | 0.528 | 9,009 |
| 30–50 eV | 0.438 | 11,449 |

**IT:** essentially non-functional (mean cos_sim ~0.01–0.04 across all CE ranges).

**Implication for sampler:** the 30–40% NCE bin contains ~16% of FT spectra (124k)
yet has median 0.714 — 0.22 below low-CE performance.  Upweighting low-CE training
examples is **not** the right fix (the model already sees these; the drop is from
increased fragmentation diversity at high CE).  The correct fix is frag-mode conditioning
(§2) so the model can predict broader, less reproducible spectra for high-CE / CID modes.

---

## 8. OOS Probability as Confidence Signal

### Finding

`pred_oos_prob` (predicted out-of-spectrum probability) is a **strong calibrated confidence
indicator**:

| pred\_oos\_prob bin | Median cos\_sim | n |
|---|---|---|
| ≤ 0.05 | **0.900** | 35,435 |
| 0.05–0.10 | 0.795 | 15,620 |
| 0.10–0.20 | 0.709 | 13,780 |
| 0.20–0.50 | 0.558 | 14,327 |
| > 0.50 | **0.261** | 6,498 |

The OOS head is well-calibrated — its score monotonically tracks actual prediction quality
across nearly a 3.5× range in median cosine sim.

### Uses

**1. Downstream reliability filter** — Flag predictions with `pred_oos_prob > 0.2` as
low-confidence in inference outputs.  Useful for library search applications.

**2. Training signal** — If OOS prediction quality degrades after model changes (e.g., CID
conditioning), check `val/cos_sim_oos_low` vs `val/cos_sim_oos_high` to catch regressions
in the OOS head specifically.

**3. Per-stratum calibration audit** — Check whether OOS is miscalibrated for specific
strata (e.g., QTOF/CID may have higher OOS scores than warranted):
```python
df.groupby(['inst_type','frag_mode'])['pred_oos_prob'].describe()
```

## 8b. Prediction-Quality Calibration Head

### Motivation

`pred_oos_prob` is already a useful confidence signal, but it is indirect: it estimates how
much probability mass lies outside the predicted fragment support.  Downstream library
search and retrieval would benefit from a direct estimate of prediction quality:

```
quality_hat ≈ E[cos_hun(predicted spectrum, true spectrum) | model outputs, metadata, DAG]
```

This score should not be treated as a new spectrum-quality metric.  It is a calibrated
reliability estimate used for:

- filtering low-confidence predictions before library export,
- reranking retrieval hits with predicted reliability,
- selecting cases for manual inspection,
- diagnosing strata where the model is accurate but underconfident, or inaccurate but
  overconfident.

### Phase 1: Posthoc Calibration (Recommended First)

Train a separate calibrator on saved validation/train predictions before adding a model
head.  This avoids disturbing FraGNNet training and tells us whether the signal is there.

Candidate inputs:

| Feature group | Examples |
|---|---|
| Existing confidence | `pred_oos_prob`, `null_formula_prob` |
| Distribution shape | formula entropy / normalized entropy, top-k mass, predicted peak count |
| DAG complexity | number of DAG nodes, number of formulae, depth reached, `force_stopped`, wrecall bucket if available for analysis |
| Metadata | `inst_type`, `frag_mode`, `prec_type`, CE / ACE / NCE, precursor m/z |
| Data source / chemistry | dataset source, ClassyFire superclass/class when available |

Target:

```python
quality_target = cos_hun  # or cos_sim_0.01 / JSS, depending on downstream use
```

Fit a small model first:

- isotonic regression on `pred_oos_prob` as the baseline,
- then LightGBM / random forest / small MLP over the feature table.

Evaluate calibration, not only correlation:

```python
df.groupby(pd.cut(df.quality_hat, [0, .2, .4, .6, .8, 1.0]))["cos_hun"].mean()
```

Keep it only if predicted-quality buckets are monotonic and each bucket's mean actual
quality is close to the predicted score.

### Phase 2: Integrated Auxiliary Head

If posthoc calibration works, add an optional head to `FraGNNetModel` that predicts one
per-spectrum scalar:

```python
pred_quality = sigmoid(quality_module(quality_features))
```

Good first input is a pooled representation, not per-fragment logits:

```
quality_features =
    concat(mol_embed_gnn_pool,
           pooled frag embedding,
           CE / ACE / NCE embedding,
           prec_type embedding,
           inst_type embedding,
           frag_mode embedding)
```

Training target should be detached from the eval metric path:

```python
quality_target = metric_output_d["cos_hun"].detach().clamp(0, 1)
quality_loss = mse(pred_quality, quality_target)
loss = loss + quality_loss_weight * quality_loss
```

Important implementation detail: keep `quality_loss` as a per-sample vector, then let the
existing batch reduction handle sample weighting.  This matches how `loss_d["loss"]` is
reduced later in `SpectrumPL._common_step`.

Suggested config:

```yaml
predict_quality: false
quality_target_metric: cos_hun
quality_loss_weight: 0.01   # sweep 0.01, 0.03, 0.05
quality_head_hidden_dim: 64
quality_loss_type: mse
```

### Evaluation

Report:

- Pearson/Spearman correlation between `pred_quality` and actual `cos_hun`
- MAE / calibration error by predicted-quality bucket
- aggregate and per-stratum calibration curves
- retrieval performance after filtering by `pred_quality > threshold`
- coverage/quality tradeoff: retained fraction vs median `cos_hun`

The head is useful even if it does not improve spectrum prediction itself, as long as it
reliably identifies predictions that should not be trusted downstream.

---

## 9. Model Ensembling

### Finding (from `frag_ann_exp.ipynb`)

Ensembling models trained with different loss weights gives **~+4% cosine sim** on the
scaffold split and **~+8–11% formula node entropy** (higher entropy = more diverse formula
predictions, better coverage of true annotated fragments).

**Evaluated on merged spectra (scaffold split) — absolute cos\_hun values (~0.65–0.68) are
lower than the per-CE 9-fold CV median (0.786) because the scaffold split uses a harder
molecule-level split; gains (+4.2%) are relative and transferable.**

**Exact numbers from notebook (4 variants × individual vs ensemble):**

| Training variant | Individual cos\_hun | Ensemble cos\_hun | Formula NE (ind) | Formula NE (ens) | NE gain |
|---|---|---|---|---|---|
| Baseline | 0.6541 | 0.6818 (+4.2%) | 0.3986 | 0.4317 | +8.3% |
| Low freq | 0.6526 | 0.6810 (+4.3%) | 0.3374 | 0.3707 | +9.9% |
| High freq | 0.6543 | 0.6817 (+4.2%) | 0.4803 | 0.5170 | +7.6% |
| **Mix** | **0.6541** | **0.6819 (+4.2%)** | **0.4057** | **0.4521** | **+11.4%** |

**Mix variant** shows the highest entropy improvement (+11.4%), meaning the most gain
in formula coverage diversity.  Cosine similarity gains are uniform across variants (~+4.2%).

**Non-isomeric (NB) variant entropy:**

| Variant | NB NE (ind) | NB NE (ens) |
|---|---|---|
| Baseline | 0.2998 | 0.3326 |
| Low | 0.2511 | 0.2803 |
| High | 0.3655 | 0.4067 |
| Mix | 0.3146 | 0.3563 |

### Strategy

**Effort: very low — no architecture changes.**  Train 3–5 seeds of the best model config,
ensemble predictions by averaging normalized intensity vectors.  The diversity across
seeds explains the gain: each model finds slightly different high-probability fragments.

Recommended: ensemble the **Mix** variant (highest entropy diversity) alongside the
Baseline for 2-model ensembles; add Low/High seeds for 4-model ensembles.

Add to Priority Order table as **Priority 9** between §6 (CID data) and §7 (Dual CE):
ensemble is zero-architecture-cost and should be validated before any structural changes.

### Follow-Up: Distill Ensemble Into a Student

The ensemble improves accuracy but multiplies inference cost.  A student model can be
trained to match the averaged ensemble distribution and recover part of the ensemble gain
while keeping single-checkpoint inference.

Teacher generation:

1. Train 3–5 teacher checkpoints using the best config seeds / loss variants.
2. Run inference over the training split and save teacher spectra per `spec_id`.
3. Average teacher predictions after normalizing each predicted sparse spectrum.
4. Store the distilled target as sparse `(mz, intensity)` arrays, preferably after the
   same binning/tolerance aggregation used by evaluation.

Student objective:

```python
loss = supervised_loss(student, true_spectrum)
loss += distill_weight * spectrum_loss(student, teacher_ensemble_spectrum)
```

Start with a conservative distillation weight:

```yaml
distill_from_ensemble: false
distill_target_fp: null
distill_weight: 0.2      # sweep 0.1, 0.2, 0.5
distill_loss_type: jss_distance_hungarian
```

Use the same sparse spectral loss family already used for training (`jss_distance_hungarian`
or `cosine_distance_hungarian`) so the student learns peak positions and intensities in the
same geometry as evaluation.  If teacher targets are binned, use the binned cosine/JSS path
consistently rather than mixing sparse and binned targets.

Why this may help:

- the ensemble target is less noisy than one teacher,
- it contains softer intensity mass over plausible fragments,
- it can improve rare peak coverage without requiring ensemble inference.

Risks / checks:

- Do not let the student ignore true labels; sweep `distill_weight`.
- Keep a validation set with no teacher leakage from test folds.
- Compare student vs best single teacher vs full ensemble on the same split.
- Track formula entropy: a student that copies only high-confidence peaks may lose the
  ensemble's diversity gain.

Success criterion:

| Model | Desired result |
|---|---|
| Best single teacher | baseline |
| Full ensemble | upper bound |
| Distilled student | closes 30–70% of the single→ensemble gap at 1× inference cost |

---

## Key Files

| File | Role |
|---|---|
| `src/fragnnet/frag/compute_frags.pyx` | Core fragmentation BFS algorithm |
| `src/fragnnet/utils/frag_utils.py` | Bond/element utilities, H-transfer logic |
| `preproc_scripts/03_prepare_dag_feats.py` | DAG preprocessing, timeout config |
| `src/fragnnet/dataset/spec_mol_frag_dataset.py` | Dataset, CE handling, charge migration |
| `src/fragnnet/pl_model/fragnnet_pl.py` | Lightning training loop |
| `src/fragnnet/model/fragnnet_model.py` | Model architecture |
| `notebooks/predictions_test_analysis.ipynb` | Source of all analysis above |

---

## References

[1] Zheng, F. et al. **PPGB-MS2: A peak-pair graph-based approach for MS/MS spectra prediction.**
    GitHub: https://github.com/zhengfj1994/PPGB_MS2
    Supplementary Figure 1 defines 28 MS/MS fragmentation reaction types (X = N, O, S; H = hydrogen)
    implemented as RDKit reaction SMARTS in `smiles_fragmentation.py`.  The 28 reactions
    (`crf1_0`–`cmf13_0`) were systematically cross-referenced against our BFS coverage in §5
    (Rearrangement Neutral Loss Pre-Pass) to identify H₂ loss and CO₂ loss as gaps in the
    fixed neutral loss list.

[2] Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018).
    **FiLM: Visual Reasoning with a General Conditioning Layer.** *AAAI 2018.*
    https://arxiv.org/abs/1709.07871
    Original paper introducing feature-wise linear modulation (FiLM): conditioning a neural
    network by predicting per-feature (γ, β) affine parameters from an auxiliary input and
    applying `y = γ · x + β` to intermediate activations.  The `CEScaler` in §3a applies
    this pattern at the output logit level, conditioned on continuous CE.

[3] Dumoulin, V., Perez, E., Schucher, N., Strub, F., de Vries, H., Courville, A., & Bengio, Y. (2018).
    **Feature-wise Transformations.** *Distill.*
    https://distill.pub/2018/feature-wise-transformations/
    Unifying treatment of feature-wise affine conditioning across domains (conditional batch
    norm, AdaIN / StyleGAN, HyperNetworks, FiLM).  Shows that multiplicative feature-wise
    conditioning is consistently more expressive than additive concatenation, motivating the
    CEScaler design over the existing `ce_insert_location` concatenation approach.
