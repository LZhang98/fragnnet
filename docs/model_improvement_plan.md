# Model Improvement Plan

Analysis based on NIST23 test set predictions (`predictions_test.pkl`, 85,660 spectra, 4,560 molecules).

---

## Current State

| Metric | Value |
|---|---|
| Median cosine sim | 0.762 |
| Mean cosine sim | 0.660 |
| cos\_sim ≥ 0.9 | 34.7% |
| cos\_sim ≥ 0.7 | 55.2% |
| cos\_sim ≥ 0.5 | 69.1% |

---

## Priority Order

| Priority | Area | Expected Impact | Effort |
|---|---|---|---|
| 1 | CID training data gap | High — 8% of test at median 0.497 | Low (data sourcing) |
| 2 | NCE/ACE as explicit conditioning input | High — strong NCE-performance curve | Medium |
| 3 | Instrument + frag\_mode conditioning | Medium — QTOF/CID domain gap | Medium |
| 4 | Data imbalancing strategy | Medium — affects all underrepresented groups | Low–Medium |
| 5 | Large molecule DAG truncation | Medium — affects MW>350 and polycyclics | High |
| 6 | Rearrangement-prone chemical classes | Medium — benzopyrans, steroids, lignans | High |
| 7 | MoNA domain normalization | Low-Medium — 22-point gap vs NIST | Low |

---

## 1. Data Imbalancing Plan

### Problem

The training set is heavily dominated along multiple axes simultaneously:

| Axis | Dominant group | Dominant % | Underrepresented groups |
|---|---|---|---|
| Instrument | FT (Orbitrap) | ~94% | QTOF (~6%), IT (~0%) |
| Frag mode | HCD | ~87% | CID (~13%) |
| Adduct | `[M+H]+` | ~66% | `[M+Na]+`, `[M+Cl]-` |
| Dataset source | NIST23 | ~91% | MoNA23 (~9%) |
| Chemical class | Benzenoids + organoheterocyclics | ~60% | steroids, lignans, benzopyrans, etc. |

These imbalances compound: a steroid measured on a QTOF instrument with CID and `[M+Na]+` sits in
the tail of every axis simultaneously.

### Strategy

#### Step 1 — Stratified batch sampling

Define a composite sampling key per spectrum:

```python
stratum_key = (inst_type, frag_mode, prec_type_bucket, chem_superclass_bucket)
```

where `prec_type_bucket` groups rare adducts (`[M+Cl]-`, `[M+Na]+`, etc.) and
`chem_superclass_bucket` groups rare chemical superclasses.

Use inverse-frequency sampling so that each stratum is sampled proportionally to
`1 / sqrt(stratum_count)` (square-root dampening avoids over-weighting tiny strata with
noisy labels).

Implementation: replace the current `DataLoader` sampler with a `WeightedRandomSampler`
using per-spectrum weights derived from the stratum frequencies.

#### Step 2 — Chemical class focal weighting in loss

For chemical classes with median cos\_sim more than 0.15 below the global median, apply a
soft focal weight to the spectrum-level loss:

```python
focal_weight = 1.0 + alpha * max(0, global_median - class_median)
```

Suggested `alpha = 2.0`.  Weight is applied per-spectrum at training time based on the
molecule's ClassyFire superclass.  Classes to target first:

| Class | Median cos\_sim | Δ from global |
|---|---|---|
| 1-benzopyrans | 0.505 | -0.257 |
| Hydrocarbon derivatives | 0.428 | -0.334 |
| Lignans / neolignans | 0.537 | -0.225 |
| Steroids | 0.595 | -0.167 |
| Stilbenes | 0.619 | -0.143 |

#### Step 3 — Oversample rare adducts

`[M+Cl]-` (n=988 in test, ~1.2%) and `[M+Na]+` (n=6,816, ~8%) are underrepresented.
Duplicate these spectra 3× and 1.5× respectively in the training dataloader.

#### Step 4 — Separate train/val splits per stratum

Ensure the validation set mirrors per-stratum composition so that improvements across all
strata are visible in aggregate val metrics.  Current splits may be dominated by FT/HCD,
masking regressions on QTOF/CID.

### Evaluation Signal

Track per-stratum median cosine similarity in the validation loop.  Log the following
groups separately to W&B:
- `val/cos_sim_FT_HCD`, `val/cos_sim_QTOF_CID`, `val/cos_sim_FT_CID`
- `val/cos_sim_Mplus_H`, `val/cos_sim_Mplus_Na`, `val/cos_sim_Mplus_Cl`
- Top-5 worst chemical classes by name

---

## 2. CID vs HCD Plan

### Problem

| Condition | Median cos\_sim | n |
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

Add `frag_mode` as a learned embedding concatenated to the molecular graph representation
before the spectrum prediction head:

```python
frag_mode_emb = nn.Embedding(num_frag_modes, emb_dim)  # HCD=0, CID=1, unknown=2
```

This allows the model to shift its predicted peak distribution based on known fragmentation
physics difference between HCD and CID (e.g., CID produces more low-mass fragment ions,
less precise m/z due to ion-trap mass accuracy).

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

## 3. NCE vs ACE Conditioning Plan

### Problem

Both normalized collision energy (NCE) and absolute collision energy (ACE) appear in the
dataset, reflecting different instrument reporting conventions:

| Coverage | Count | % |
|---|---|---|
| Both NCE and ACE | 62,333 | 72.8% |
| Only NCE | 15,865 | 18.5% |
| Only ACE (no NCE) | 7,462 | 8.7% |

ACE-only spectra are predominantly QTOF (4,783 / 7,462) and show severe performance
degradation at high ACE:

| ACE bin (eV) | Median cos\_sim | n |
|---|---|---|
| 0–10 | 0.863 | 1,555 |
| 10–20 | 0.653 | 1,087 |
| 20–30 | 0.558 | 1,176 |
| 30–50 | 0.367 | 2,723 |
| 50–100 | 0.269 | 920 |

NCE and ACE are related by `NCE ≈ ACE / (prec_mz / 500)` but the real-data correlation is
0.779 (not 1.0), indicating inconsistent normalization conventions across instruments.

### Plan

#### Step 1 — Unified collision energy representation

Compute a unified `nce_unified` feature at preprocessing time:

```python
if nce is not None:
    nce_unified = nce
elif ace is not None and prec_mz is not None:
    nce_unified = ace / (prec_mz / 500.0)
else:
    nce_unified = NaN  # handled separately
```

Store `nce_unified` and `ce_source` (one of `"nce"`, `"ace_converted"`, `"missing"`) as
features in the processed spectrum dataframe.

#### Step 2 — NCE as continuous model input

NCE is currently likely binned or ignored.  Replace with a continuous Fourier feature
encoding (sinusoidal, similar to positional encoding):

```python
nce_feat = fourier_encode(nce_unified / 100.0, num_freqs=16)  # normalized to ~[0,1]
```

Concatenate to the precursor/spectrum-level representation before the prediction head.

The NCE-performance curve is steep (median drops from 0.935 at NCE 0–20 to 0.646 at NCE
120–200), which means correctly conditioning on NCE can substantially improve high-energy
predictions.

#### Step 3 — Handle missing CE gracefully

For the ~0% of spectra with no CE (currently zero in test, but possible in inference),
learn a marginal CE distribution and use it as a soft prior rather than zeroing the feature.
A `ce_missing` flag embedding should replace `nce_feat = 0` to avoid conflating "zero
energy" with "unknown energy."

#### Step 4 — ACE conversion validation

For spectra where both ACE and NCE are available (n=62,333), validate the conversion
formula per instrument type:

```python
# check residual: actual_nce vs ace_converted
residual = df.groupby('inst_type')[['nce', 'nce_from_ace']].apply(
    lambda g: (g['nce'] - g['nce_from_ace']).abs().median()
)
```

If QTOF uses a different normalization constant than FT (possible — some QTOF vendors use
a different reference mass), use instrument-specific conversion coefficients learned from
the training data.

#### Step 5 — ACE-only QTOF spectra

The 4,783 QTOF ACE-only spectra are systematically the worst (median 0.367 at ACE 30–50 eV).
Prioritize these for CID-specific conditioning (§2) and instrument-conditioning (instrument
embedding) since ACE and QTOF co-occur.

---

## 4. Large Molecule Plan

### Problem

Performance degrades sharply with molecular size:

- MW > 350 Da: substantial performance drop
- num\_atoms > 30: consistent degradation

Three distinct failure modes:

**A. DAG truncation** — combinatorial explosion of bond-break paths triggers the 150s
timeout.  When `force_stop=True`, the DAG is pruned to `reached_depth - 1`, losing deep
fragment coverage.

**B. Formula redundancy** — large molecules generate many fragments with identical
molecular formulas but different structures; the formula-matching signal becomes noisy.

**C. Rearrangement-driven fragmentation** — some large molecule classes (steroids,
terpenoids) fragment via retro-Diels-Alder, ring-opening, and isoprene loss — mechanisms
not present in the bond-cleavage graph.

### Plan

#### Step 1 — Audit `force_stopped` rates per chemical class

```python
dag_stats = load_dag_hdf5_stats()  # force_stopped, reached_depth, dag_num_nodes
df_audit = df.merge(dag_stats, on='mol_id')
print(df_audit.groupby('klass')[['force_stopped', 'dag_num_nodes', 'cos_sim_0.01']].agg(['mean','median']))
```

This separates classes where graph truncation is the dominant cause of bad performance from
classes with genuine model limitations.

#### Step 2 — Prioritized BFS in fragmentation

Replace the current uniform-bond BFS with a bond-priority queue.  Score bonds by
fragmentation likelihood before enumeration:

- Heteroatom-adjacent bonds: higher priority (N–C, O–C, S–C)
- Benzylic / allylic positions: higher priority
- Ring bonds: lower priority unless labile (glycosidic, ester)
- Simple C–C in aliphatic chains: lowest priority

This ensures the most chemically meaningful fragments are generated first and survive even
if timeout is reached.

#### Step 3 — Expose `force_stopped` and `dag_num_nodes` as model inputs

Add these as scalar features to the molecule-level representation:

```python
mol_features = concat([existing_features, log(dag_num_nodes + 1), force_stopped_flag])
```

Let the model learn to reduce confidence / shift predictions when it knows the DAG is
truncated.

#### Step 4 — Neutral loss DAG representation

For large molecules, represent the DAG in terms of **neutral losses** (delta formula from
parent) rather than absolute fragment formulas.  Losses are more structurally specific than
absolute formulas at high MW and reduce formula redundancy.

#### Step 5 — MAGMA fallback for truncated DAGs

For molecules where `force_stop=True` or `dag_num_nodes < threshold`, run MAGMA to generate
a chemically-prioritized fragment tree as a supplement.  MAGMA uses thermodynamic scoring
to rank fragmentation pathways and is better suited to polycyclic structures.

#### Step 6 — 128-atom hard limit audit

Verify what fraction of training/test molecules exceed 128 atoms.  If non-trivial, extend
`MAX_NUM_NODES` and `MAX_NUM_EDGES` in `compute_frags.pyx` and reprocess affected molecules.

---

## 5. Fragmentation Graph Limitations by Chemical Class

The algorithm is pure bond-cleavage with BFS.  It has **no rearrangement rules, no ring
priority, no thermodynamic scoring**.  Some classes are fundamentally mispredicted because
their real MS2 fragmentation proceeds via mechanisms outside the model:

### Classes with structural fragmentation graph failures

**1-benzopyrans (median 0.505) — retro-Diels-Alder (RDA)**
- Flavonoids and isoflavonoids (classified here) fragment via **RDA at the C-ring**,
  producing A-ring and B-ring ions with specific diagnostic m/z.
- This is a rearrangement; simple bond cleavage generates different fragments.
- Fix: implement RDA as an explicit fragmentation rule for 6-membered rings with the
  [C=C–C=O] pattern.

**Steroids and steroid derivatives (median 0.595) — RDA + ring junction opening**
- Dominant fragmentation: RDA on ring A/B or B/C junction, loss of side chains via
  specific ring-opening.
- 4-fused-ring system also causes combinatorial explosion → likely high `force_stopped` rate.
- Fix: RDA rule + prioritized ring-junction bond scoring.

**Lignans / neolignans (median 0.537) — benzyl-ether cleavage + furan ring-opening**
- Involves preferential benzyl-ether bond cleavage (not prioritized by uniform BFS),
  ring-opening of furanoid rings, and retro-[2+2] in some subtypes.
- Fix: add benzyl-ether bond priority score; furan ring-opening rule.

**Prenol lipids / terpenoids (median 0.634) — isoprene loss**
- Characteristic fragmentation via loss of isoprene units (C₅H₈, 68 Da) after
  cyclization/retro-cyclization.
- The DAG can generate C5 losses via random bond cleavage but doesn't preferentially
  weight isoprene-unit boundaries.
- Fix: detect isoprene-unit junctions by SMARTS and upweight those bonds.

### Classes likely suffering from timeout truncation

- **Hydrocarbon derivatives (median 0.428):** saturated polycyclics (adamantane, hopanes)
  have exponential ring-bond combinations; BFS explodes before reaching terminal fragments.
- **Organic polymers (median 0.680):** repeated unit × fragmentation depth = combinatorial
  explosion; many also exceed the 128-atom limit.

### Test to confirm: `force_stopped` cross-tabulation

| If... | Then root cause is... | Fix |
|---|---|---|
| `force_stopped` rate high + low cos\_sim | DAG truncation | Prioritized BFS, MAGMA fallback |
| `force_stopped` rate low + low cos\_sim + RDA chemistry | Missing rearrangement | Explicit RDA/rearrangement rules |
| `force_stopped` rate low + low cos\_sim + no rearrangement | Training data imbalance | Weighted sampling, more data |

---

## 6. MoNA Domain Gap

MoNA23 performance (median 0.559) vs NIST23 (median ~0.782) is inflated by IT spectra
(all 35 IT samples are MoNA23-only, median 0.000).  Strip IT from MoNA before drawing
conclusions:

```python
mona_ft = df[(df['dset'] == 'mona23') & (df['inst_type'] == 'FT')]
print(mona_ft['cos_sim_0.01'].median())  # expected to be closer to NIST23 FT
```

If the FT-only MoNA gap is small, the MoNA issue is provenance (IT data) not curation.
If a gap remains, investigate peak normalization and intensity preprocessing differences
between MoNA and NIST curation pipelines.

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
