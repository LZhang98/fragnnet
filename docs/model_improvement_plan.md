# Model Improvement Plan

Analysis based on NIST23 test set predictions, all 9 CV folds combined (764,637 spectra, 28,256 unique molecules).

---

## Current State

**Overall (764,637 spectra, 28,256 unique molecules — NIST23 9-fold CV)**

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
| NCE conditioning | ✅ enabled | `nce: True`, `ce_insert_location: mlp` |
| Instrument type conditioning | ✅ enabled | `inst_type: True`, `inst_insert_location: mlp` |
| Precursor type conditioning | ✅ enabled | `prec_insert_location: mlp` |
| ACE → NCE conversion for QTOF | ✅ done | `fill_missing_nce()` via `ace_to_nce()` |
| FT / QTOF / IT in inst\_types | ✅ included | `inst_types: [FT, QTOF, IT]` |
| **Fragmentation mode conditioning** | ❌ **missing** | `frag_mode` not in config or model |
| **Dual CE input (ACE + NCE)** | ❌ **missing** | assert prevents both from being loaded |

The 0.762 median is with all currently-implemented conditioning active.  The two missing
conditioning signals are architecture gaps that gate the FT/CID and QTOF improvements.

---

## Priority Order

| Priority | Area | Expected Impact | Effort |
|---|---|---|---|
| 1 | `force_stopped` + formula-recall (AR) diagnostic per class | Unblocks §4 and §5 decisions | Very low (query) |
| 2 | **Frag-mode conditioning** (HCD/CID token) | High — FT/CID gap 0.785→0.497, model blind to it | Low (arch, same pattern as inst\_type) |
| 3 | QTOF/CID upweighting in sampler | High — sampler may not weight by inst/frag\_mode | Low |
| 4 | Chemical class focal loss weighting | Medium — size explains only 4.2% of variance | Low–Medium |
| 5 | Prioritized BFS for polycyclic DAG truncation | High — anthracenes 0.276, AR=0.81 vs GrAFF 0.95 | Medium (Cython) |
| 6 | CID training data sourcing | High — FT/CID median 0.497 with ~8% of data | Low (data work) |
| 7 | Dual CE input (ACE + NCE, no conversion) | Medium — eliminates 0.779-correlation noise for QTOF | Low–Medium (small arch change) |
| 8 | MoNA FT-only gap check | Low — 2-line diagnostic | Very low (query) |
| 9 | **Model ensembling** | +4% cos\_hun, no arch change needed | Very low (multi-seed inference) |
| 10 | OOS calibration per stratum | Confidence filter for downstream use | Very low (query) |

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

#### Step 1 — Check existing sampler before changing anything

The current config uses `dynamic_sampler`.  Verify whether it already weights by
`inst_type` / `frag_mode` before adding new weights:

```bash
grep -rn "dynamic_sampler\|WeightedRandom\|inst_type\|frag_mode" \
    src/fragnnet/pl_model/ src/fragnnet/dataset/
```

Only proceed to Step 2 if the sampler does not upweight QTOF/CID.

#### Step 2 — Add QTOF/CID upweighting to sampler

Add stratum weights based on `(inst_type, frag_mode)` using inverse-sqrt frequency
dampening.  This is a `DataLoader` change only — no model or config changes required:

```python
stratum_key = (inst_type, frag_mode, prec_type_bucket, chem_superclass_bucket)
# weight = 1 / sqrt(stratum_count)
```

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
The current dataset enforces mutual exclusivity:

```python
# base_dataset.py:142
assert not (spec_params["ace"] and spec_params["nce"])
```

CE coverage in test set:

| Coverage | Count | % |
|---|---|---|
| Both NCE and ACE | 62,333 | 72.8% |
| Only NCE | 15,865 | 18.5% |
| Only ACE (QTOF-heavy) | 7,462 | 8.7% |

### Proposed Approach: Feed Both, No Conversion

Pass ACE and NCE as **two separate scalar inputs** with explicit validity masks.  The
model embeds each independently and concatenates them.  When one is missing, a learned
`[MISSING]` token is used instead.  This preserves full information and lets the model
learn instrument-specific reliability of each CE type from data.

| Spectrum | ace | nce | What model sees |
|---|---|---|---|
| FT/HCD | NaN or real | real | `[missing_ace, nce_embed]` |
| QTOF/CID ACE-only | real | NaN | `[ace_embed, missing_nce]` |
| Both available | real | real | `[ace_embed, nce_embed]` |

### Implementation

**1. Dataset** (`base_dataset.py`):
- Remove the `assert not (ace and nce)` guard at line 142.
- Load both columns unconditionally; replace NaN with sentinel `-1`.
- Store as `spec_ace` and `spec_nce` with `spec_ace_mask` / `spec_nce_mask` boolean
  tensors indicating valid vs missing.

**2. Model** (`base_model.py`):
- Add a `dual_ce` path alongside existing `CEModel`.
- Embed each CE with its own `CEModel` instance (independent weights — ACE and NCE
  have different scales and units).
- Concatenate: `ce_embed = concat([ace_embed, nce_embed])` → size `2 × ce_insert_size`.
- Where `spec_ace_mask=False`, replace `ace_embed` with a learned `ace_missing_token`;
  same for NCE.

**3. Config**:
```yaml
# replaces:  nce: True
dual_ce: True
ce_insert_size: 32      # per-CE; concat output = 64
ce_insert_location: mlp
ce_insert_type: embed
```

### Why No Conversion Is Better

- ACE (eV) and NCE (%) encode physically different quantities — absolute beam energy vs
  mass-normalised energy.  They carry partially independent information (r=0.779).
- QTOF natively reports ACE; converting to NCE via `prec_mz` adds m/z-dependent noise.
- FT natively reports NCE; it already has reliable NCE — the ACE path is absent or noisy.
- The model implicitly learns "trust ACE for QTOF, trust NCE for FT" from training data
  without any hand-coded conversion coefficients or per-instrument scaling factors.

---

## 4. DAG Truncation for Large / Polycyclic Molecules

### Constraint

All fixes must be **structure-only** — they must work at inference time from SMILES alone,
with no observed spectrum.  MAGMA is not usable here because it scores fragments against
the observed spectrum, which is unavailable at inference.

### Formula Recall Baseline (from `formula_ann_exp.ipynb`)

Comparison of formula-level annotation recall (AR) and precision (AP) across models,
measured against ground-truth MS/MS annotations from the NIST23 scaffold split:

| Model | True Recall (AR) | Weighted Recall (AWR) | Pred Precision (AP) | Weighted Precision (AWP) | F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| FraGNNet-D4 | 0.81 | 0.90 | 0.98 | 1.00 | 0.89 | 0.95 |
| GrAFF | **0.95** | **0.98** | 0.61 | 0.80 | 0.74 | 0.88 |
| ICEBERG | 0.66 | 0.81 | 0.99 | — | 0.79 | — |

**FraGNNet-D4 misses ~19% of ground-truth annotated fragment formulas** (AR=0.81 vs GrAFF
AR=0.95). GrAFF's higher recall comes at the cost of precision (AP=0.61 vs 0.98) — it
over-generates fragments.  The target for improved BFS is **AR > 0.90 while keeping AP
> 0.95**.  Formula-recall improvement is the primary quantitative goal of §4.

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

#### Step 3 — Ring-contraction strategy for polycyclics

For molecules with ≥ 3 fused rings, add an optional fragmentation pass that treats each
fused ring system as an atomic unit (does not break internal ring bonds at depth 1).
This dramatically reduces the BFS branching factor while still enumerating all fragments
that result from breaking bonds at ring junctions or on external chains.  Inference-safe:
ring system detection is structure-only.

#### Step 4 — Expose `force_stopped` and `dag_num_nodes` as model inputs

```python
mol_features = concat([existing_features, log(dag_num_nodes + 1), force_stopped_flag])
```

The model learns to adjust confidence when it knows the DAG is incomplete.  Both values
are computed at preprocessing time and available at inference.

#### Step 5 — Neutral loss representation for high-MW molecules

For molecules above a MW threshold, represent DAG edges as neutral losses (Δformula from
parent) rather than absolute fragment formulas.  Neutral losses are more structurally
specific than absolute formulas at high MW and reduce formula-redundancy noise.

#### Step 6 — 128-atom hard limit audit

Check what fraction of training/test molecules hit the `MAX_NUM_NODES=128` assertion in
`compute_frags.pyx`.  If non-trivial, extend the limit and reprocess.

---

## 5. Chemical Class Fragmentation Failures

### Revised Understanding of RDA and Rearrangements

RDA and other rearrangement products are **not missing from the DAG**.  For a 6-membered
ring RDA (e.g. flavonoid C-ring), breaking two bonds sequentially reaches the same product
masses at depth ≤ 2, well within `max_depth=4`.  The fragment *masses* are present.

The actual problem is **intensity misprediction**: among many competing depth-2 fragments
that share the same formula, the model hasn't learned to assign high intensity to the
specific fragments that correspond to preferred fragmentation pathways (RDA, benzyl-ether
cleavage, isoprene loss).  This is a training signal problem, not a graph coverage problem.

Consequence: explicit rearrangement rules in the fragmentation algorithm are **not
needed** for mass coverage.  They may still help by surfacing RDA fragments earlier in the
BFS so they receive better graph attention, but this is a secondary concern.

### Diagnosis Decision Tree

Run the `force_stopped` audit (§4 Step 1) first, then apply:

| `force_stopped` rate | Cos\_sim | Chemistry | Root cause | Fix |
|---|---|---|---|---|
| High (>30%) | Low | Any | DAG truncation | Prioritized BFS (§4 Step 2), ring-contraction (§4 Step 3) |
| Low | Low | RDA / ring-strain | Intensity misprediction | Focal loss (§1 Step 3), bond-priority BFS |
| Low | Low | No special chemistry | Data imbalance | Focal loss + more training data |

### Classes and Their Expected Root Cause

| Class | Median | Residual | Expected cause |
|---|---|---|---|
| Anthracenes | 0.276 | -0.439 | DAG truncation (3 fused rings) |
| Morphinans | 0.380 | -0.299 | DAG truncation (polycyclic alkaloid) |
| Furans | 0.383 | -0.313 | Intensity misprediction — ring-opening bonds not prioritised |
| Lactones | 0.416 | -0.281 | Intensity misprediction — ring-strain cleavage not prioritised |
| 2-arylbenzofuran flavonoids | 0.360 | -0.318 | Intensity misprediction — benzofuran + RDA pattern |
| Benzopyrans | 0.511 | -0.221 | Intensity misprediction — RDA masses present, weights wrong |
| Steroids | 0.595 | -0.152 | Mixed — DAG truncation + intensity misprediction |
| Prenol lipids | 0.634 | -0.127 | Intensity misprediction — isoprene junctions not prioritised |
| Lignans / neolignans | 0.537 | — | Intensity misprediction — benzyl-ether cleavage not prioritised |
| **Stilbenes** | **0.619** | — | Intensity misprediction — trans/cis stilbene retro-Diels-Alder |
| **Hydrocarbon derivatives** (superclass) | **0.428** | — | DAG priority failure — few heteroatom anchors, BFS gains little structural signal |

### Bond-Priority BFS as the Unified Fix

The prioritized BFS (§4 Step 2) addresses both truncation and intensity misprediction:
- Truncation: chemically meaningful fragments are enumerated before timeout
- Intensity: BFS order influences graph depth embedding — fragments reached earlier
  get shallower depth features, and training will associate those features with higher
  predicted intensity if the real spectra confirm it

Specific SMARTS patterns to detect and upweight for intensity misprediction classes:

| Class | High-priority bond pattern |
|---|---|
| Benzopyrans / flavonoids | C-ring junction bonds: `[c,C]-[O,C]` in 6-membered O-rings |
| Lactones | Ester C–O bond: `[C](=O)-O-[C]` |
| Furans | α-bond to furan oxygen: `[c]-[O]-[c]` ring bond |
| Prenol lipids | Isoprene junction: `[C](=C)-[C]-[C]=C` |
| Lignans | Benzyl-ether: `[c]-[CH2]-O` |

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
| MoNA FT-only check | `df[(df.dset=='mona23')&(df.inst_type=='FT')].cos_sim_0.01.median()` | §6 action |
| ACE→NCE residual per instrument | `both.groupby('inst_type').apply(lambda g: (g.nce - g.nce_converted).abs().median())` | §3 Step 1 |
| Current sampler weighting | `grep -rn "dynamic_sampler\|WeightedRandom" src/fragnnet/` | §1 Step 1 |
| OOS calibration per stratum | `df.groupby(['inst_type','frag_mode'])['pred_oos_prob'].describe()` | §8 confidence filter |

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

---

## 9. Model Ensembling

### Finding (from `frag_ann_exp.ipynb`)

Ensembling models trained with different loss weights gives **~+4% cosine sim** on the
scaffold split (cos\_hun: individual 0.654 → ensemble 0.682) and **~+8% formula node
entropy** (individual 0.399 → ensemble 0.432).  Higher entropy indicates more diverse
formula predictions — better coverage of true annotated fragments.

| Training variant | Individual cos\_hun | Ensemble cos\_hun | Formula NE (ind) | Formula NE (ens) |
|---|---|---|---|---|
| Baseline | 0.654 | 0.682 (+4.3%) | 0.399 | 0.432 (+8.3%) |
| Low weight | — | — | lower NE → less diverse | — |
| High weight | — | — | higher NE → more diverse | — |
| Mix | — | — | balanced | — |

### Strategy

**Effort: very low — no architecture changes.**  Train 3–5 seeds of the best model config,
ensemble predictions by averaging normalized intensity vectors.  The diversity across
seeds explains the gain: each model finds slightly different high-probability fragments.

Add to Priority Order table as **Priority 9** between §6 (CID data) and §7 (Dual CE):
ensemble is zero-architecture-cost and should be validated before any structural changes.

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
