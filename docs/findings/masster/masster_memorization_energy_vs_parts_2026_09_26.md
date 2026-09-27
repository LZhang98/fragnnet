# Is MASSter's memorized allocation keyed on energy, on molecules or on parts?

> **Status:** FINDING · **Updated:** 2026-09-26
> Inference only, on `3bej6fq7` (f0, unmerged `[M+H]+`/FT, the route map's checkpoint).
>
> - **Memorization is per molecule, not per energy.** An NCE shift of ±1 in the intensity head
>   costs train and val alike (damage ratio 1.04-1.11, no gap closed). Removing CE costs train
>   1.37x what it costs val, with an excess gap closed of 0.045. Global attention's is 0.12-0.13.
> - **Memorization is not per part.** Train fit is flat across molecule novelty (0.909-0.931,
>   Spearman 0.02). Within one spectrum, peaks from breaks seen in no other train molecule are
>   fit almost as well as common ones (error +0.026).
> - **Generalization is per part.** On val the same within-spectrum contrast is +0.097, 3.7 times
>   train's. It holds at fixed molecule similarity. 28% of val fragment-peak intensity sits on
>   breaks no train molecule has, and val captures 0.49 of it against 0.69 on common breaks.
> - **The ±1 damage (0.06-0.09 on both splits) is the categorical NCE code, not memorization.**
>   Shifts that land on an NCE absent from training cost 0.07-0.12, and shifts onto a trained
>   value cost 0.014-0.04. The code is diagnosed in the
>   [QET projection finding](masster_qet_projection_candidate_union_2026_09_26.md#the-nce-input-is-categorical).
> - **The heavier side of a cut is the more memorizing one, but not the key.** Removing its
>   identity costs train 1.78x what it costs val, and 2.8x at NCE ≤ 20. The lighter side, often
>   the neutral loss, scores 1.42x and carries 2.5 times more of what val uses. Its excess gap
>   closed (0.041) is a third of global attention's
>   ([side probe](#side-of-cut-probe-2026-09-26)).
> - **The allocator's set message is the most train-specific route measured.** Replacing the
>   message each candidate receives from its candidate set with its spectrum's mean costs train
>   3.37x what it costs val. Val barely uses it (0.023), and only on close analogues
>   ([set-message probe](#allocator-set-message-probe-2026-09-26)).
> - **NCE values are not interchangeable across the two acquisition ladders.** On molecules
>   measured on both, spectra at adjacent values of the two ladders agree at 0.71-0.84. Same-ladder
>   spectra 10 NCE apart agree at 0.84.

## Question

The [route map](masster_route_map_2026_09_24.md) found that the lookup is keyed on molecule
identity, written by global attention and rebuilt by local message passing when that is removed
([Step 1](../../plans/masster/masster_memorization_route_plan.md#eval-mode-trainval-gap-on-the-step-1-checkpoints-2026-09-25)).
Two questions were open:

1. **Energy.** The encoder is CE-blind, so the key is identical across a molecule's ~12
   spectra. CE reaches the allocation only through the 128-column context in the candidate token.
   Does the allocator store a separate allocation per energy?
2. **Parts.** Is train fit a whole-molecule lookup, or does it lean on substructures shared with
   other train molecules?

## Method

Scripts and outputs: `output/memorization_parts_ev_2026_09_26/` (local, Git-ignored).
`run_gpu.sh` ran both GPU passes on the local RTX 2070 SUPER (19 min and 3 min).

### Energy: the `ctx` scope of the route-map probe

`scripts/debug/debug_pooled_embedding_memorization.py` gains a `ctx` scope and a
`--per_spectrum_fp` dump. Tests: `tests/test_debug_pooled_embedding_memorization.py`, 20 pass.

- **What moves.** Only the context rows the intensity head reads. The proposal keeps the exact
  context, so bond scores and the candidate set do not move; a unit test checks that
  `bond_break_logits` and `pred_mzs` are bit-identical. With `proposal_selection_only` the
  proposal's CE-dependent log-probabilities never reach the head, so the context token is the
  allocation's only CE input.
- **Conditions.**
  - `ctx_popmean`: every spectrum gets the batch mean over spectra, so CE is removed.
  - `ctx_shift_m5`, `_m1`, `_p1`, `_p5`: the context re-encoded at NCE + delta, clamped to
    [1, 200].
  - The route map's `cc_molmean`/`cc_shuffle` references and the three `gsa` modes run alongside.
- **Sample.** The route map's: 60 batches of 512 per split, eval mode, the same batches for every
  condition. The baseline and the `gsa` rows reproduce the route map to 0.001.
- **Reads.** The damage ratio is train drop over val drop. A route both splits use alike scores
  near train/val cosine (about 1.25; the pools score 1.21-1.25). Excess gap closed is measured
  against the pools at matched val damage, as in the route map.
- **Real drift.** For comparison: cosine between two spectra of one train molecule by NCE gap,
  on 3,000 train molecules, using 0.01 Da bins and plain cosine as the metric does.

### Parts: per-peak error by cut-bond environment rarity

- **Predictions.** `collect_parts.py` stores the evaluated prediction, target and kept fragments
  with their atom sets for every spectrum of 2,500 seeded train molecules (30,777 spectra) and
  2,500 val molecules (30,608). Mean `cos_sim_0.01` is 0.9208 and 0.7352.
- **Part key.** For every bond, the key is its type, its ring flag and the unordered pair of its
  endpoints' Weisfeiler-Lehman atom identifiers at radius r = 1 or 2 (Morgan-style, no
  deduplication).
- **Frequency.** The number of *other* train molecules containing that bond anywhere, over all
  12,668 train molecules. A train molecule does not count itself.
- **Fragment rarity.** The minimum frequency over the fragment's cut bonds: its least familiar
  break.
- **Peak rarity.** A target peak (0.01 Da bin) is explained by the kept fragments with a candidate
  ion in its bin. Its rarity is the maximum over them, the most familiar explanation, so a peak
  counts as rare only when every explanation is rare.
  - Peaks explained only by the precursor (16.8-16.9% of target intensity) are left out.
  - So are uncovered peaks (8.3-8.5%).
- **Error.** `|log10((p + 0.001) / (t + 0.001))|` on sum-normalized binned intensities, weighted
  by target share t. "Captured" is `Σ min(p, t) / Σ t`.
- **Within-spectrum contrast.** Per spectrum, the weighted error of rare peaks (≤ 2 other
  molecules) minus that of common peaks (≥ 10), averaged over spectra that have both. This holds
  the molecule, the energy and the spectrum's overall fit fixed.
- Scripts: `parts_analysis.py` (log `parts_analysis_log.txt`) and `probe_analysis.py` (log
  `probe_analysis_log.txt`).

## Result: energy

Cosine is `cos_sim_0.01`, eval mode.

| condition | train | val | gap | train drop | val drop | damage ratio | excess gap closed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | 0.9242 | 0.7437 | 0.1806 | — | — | — | — |
| `ctx_popmean` | 0.5471 | 0.4686 | 0.0785 | 0.377 | 0.275 | 1.37 | 0.045 |
| `ctx_shift_m1` | 0.8561 | 0.6822 | 0.1739 | 0.068 | 0.062 | 1.11 | −0.006 |
| `ctx_shift_p1` | 0.8303 | 0.6533 | 0.1770 | 0.094 | 0.090 | 1.04 | −0.015 |
| `ctx_shift_m5` | 0.8845 | 0.7166 | 0.1679 | 0.040 | 0.027 | 1.47 | 0.007 |
| `ctx_shift_p5` | 0.8680 | 0.7021 | 0.1659 | 0.056 | 0.042 | 1.35 | 0.006 |
| `gsa_popmean` | 0.6929 | 0.6540 | 0.0389 | 0.231 | 0.090 | 2.58 | 0.123 |
| `gsa_xmol` | 0.6377 | 0.6163 | 0.0215 | 0.287 | 0.127 | 2.25 | 0.133 |
| `cc_molmean` | 0.5312 | 0.4184 | 0.1128 | 0.393 | 0.325 | 1.21 | reference |
| `cc_shuffle` | 0.4310 | 0.3481 | 0.0829 | 0.493 | 0.396 | 1.25 | reference |

Real spectra of one train molecule agree at 0.946 at the same NCE (1,091 pairs), 0.936 at a
5-NCE gap (12,868) and 0.839 at 10 (14,297). Only 13 pairs are 1-2 NCE apart.

Shift damage depends on where the shift lands (train / val):

| shift lands on | ±1 | ±5 |
| --- | --- | --- |
| an NCE absent from training | 0.076-0.089 / 0.068-0.079 | 0.083-0.116 / 0.113-0.116 |
| an NCE with ≥ 1,000 train spectra | 0.018 / 0.020 (−1 only) | 0.029-0.040 / 0.014-0.036 |

+1 onto a common value is left out. It is dominated by 75 → 76, and 76 shares its input code
with 180 ([collision](masster_qet_projection_candidate_union_2026_09_26.md#the-nce-input-is-categorical)).

`ctx_popmean` by NCE bin (train drop / val drop): ≤ 20: 0.647 / 0.535; 20-40: 0.241 / 0.197;
40-60: 0.026 / 0.027; 60-100: 0.320 / 0.187; 100-200: 0.680 / 0.471. The batch-mean context sits
near mid-range NCE, so that bin barely moves.

Cross-ladder spectra, all splits. The ladders are labelled as in the QET finding: A is
10, 15, … 60, 75, 90, …; B is 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 76, 88, 100, ….

| B value vs A value | molecules with both | mean cosine |
| --- | ---: | ---: |
| 76 vs 75 | 23 | 0.753 |
| 44 vs 45 | 25 | 0.836 |
| 14 vs 15 | 25 | 0.815 |
| 88 vs 90 | 22 | 0.782 |
| 9 vs 10 | 12 | 0.742 |
| 27 vs 25 | 13 | 0.708 |
| 54 vs 55 | 7 | 0.705 |

535 of 22,411 `[M+H]+`/FT molecules have values from both ladders.

## Result: molecules and parts

Molecule novelty is the max Tanimoto (Morgan r2, 2,048 bits) to another train molecule. For
train molecules the molecule itself is excluded.

| novelty | train `none` | val `none` | train `gsa_popmean` damage | val `gsa_popmean` damage |
| --- | ---: | ---: | ---: | ---: |
| ≤ 0.4 | 0.917 | 0.609 | 0.252 | 0.028 |
| 0.4-0.5 | 0.925 | 0.661 | 0.246 | 0.049 |
| 0.5-0.6 | 0.929 | 0.725 | 0.229 | 0.081 |
| 0.6-0.7 | 0.931 | 0.773 | 0.226 | 0.105 |
| 0.7-0.8 | 0.921 | 0.827 | 0.219 | 0.129 |
| > 0.8 | 0.909 | 0.838 | 0.211 | 0.131 |

Spearman with novelty: train fit 0.022, val fit 0.287.

Per-peak error by break rarity at radius 2. The count is other train molecules with the break;
the share is of fragment-explained target intensity.

| rarity | train share | val share | train error | val error | train captured | val captured |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.296 | 0.277 | 0.159 | 0.627 | 0.836 | 0.488 |
| 1-2 | 0.177 | 0.189 | 0.141 | 0.460 | 0.853 | 0.602 |
| 3-9 | 0.203 | 0.202 | 0.124 | 0.414 | 0.876 | 0.642 |
| 10-99 | 0.270 | 0.281 | 0.111 | 0.337 | 0.891 | 0.699 |
| 100+ | 0.054 | 0.051 | 0.112 | 0.330 | 0.894 | 0.690 |

Within-spectrum contrast, rare minus common (SE):

| radius | train | val | spectra (train; val) |
| --- | --- | --- | --- |
| 2 | +0.026 (0.002) | +0.097 (0.005) | 11,925; 11,747 |
| 1 | +0.061 (0.003) | +0.081 (0.006) | 8,429; 7,670 |

The val gradient holds inside each molecule-novelty bin at radius 2: 0.494 → 0.236 above 0.7,
0.538 → 0.370 at 0.5-0.7, and 0.677 → 0.536 at or below 0.5. It also holds inside each peak-size
bin: for peaks with t ≥ 0.1, train 0.094 → 0.078 and val 0.660 → 0.309.

## Reading

- **No per-energy key.** A 1-NCE change should be nearly free for a CE-smooth model, and a
  per-energy lookup would make it cost train far more than val. It costs both about the same.
  Removing CE is mildly train-specific (1.37x, excess 0.045, about a third of global attention's).
  That excess concentrates at NCE above 60 (1.44-1.71x). A molecule's high-energy allocation is
  somewhat more train-specific than its low-energy one, which fits the finding that run-to-run
  disagreement grows with NCE ([variance](masster_variance_novelty_2026_09_24.md)).
- **The energy input is the weak point, not memorization through it.** Off-grid energies cost
  0.07-0.12 on both splits, and on-grid shifts cost about what real spectra change. This is the
  categorical code of the QET finding seen from the other side.
- **The ladders need their own input.** Adjacent values of the two ladders differ by 0.71-0.84
  cosine on the same molecule. So a shared NCE axis does not describe both acquisitions. The
  categorical code lets the model separate the ladders through the exact integer, and a smooth
  energy input would take that away unless the ladder enters explicitly. Resolution and collision
  gas do not separate them (both `4`, `N2`). The per-ladder counts are small (7-25 molecules).
- **Train fit is a whole-molecule lookup.** Unique train molecules are fit as well as molecules
  with close neighbours. A peak from a break no other train molecule has is fit at 0.836 captured
  against 0.894 for common breaks. A store of per-part responses would instead fit unique parts
  *better* on train, since each is fitted to one example. The small residual gradient (+0.026) is
  about what harder chemistry costs.
- **Val transfer runs through parts.** The same contrast is 3.7 times larger on val, within the
  spectrum and at fixed molecule similarity. What reaches a new molecule is familiarity with its
  breaks. The molecule-level store adds little there, except on near-analogues: global
  attention's val damage grows from 0.028 to 0.131 with similarity, so the lookup helps on close
  analogues and gives almost nothing to novel molecules.
- **Where val loses intensity.** 28% of val fragment-peak intensity is on breaks no train
  molecule has (radius 2), captured at 0.49.

## What this changes

- **Regularize molecule identity broadly.** The key is molecule-level and not energy- or
  part-level, so this is the target. `k1` key dropout and the global-message arms (`gm1`, `gm2`)
  are the right family, as are more molecules that cover new break environments. Blocking one
  channel is not ([plan](../../plans/masster/masster_memorization_route_plan.md)).
- **Break-local evidence is what transfers.** A signal keyed on cut-bond environment, such as
  the [bond-local learner](masster_bond_local_learner_2026_09_24.md) (gate grey zone), targets
  the part of the model that generalizes. Its value is on unseen-break peaks.
- **A smooth energy input needs a ladder input next to it.** Otherwise `ce_insert_type: id` or
  `int_embedder: rbf` merges two acquisitions that differ at the same nominal NCE. The CE-blind
  candidate token with a shared β(NCE)
  ([breakdown curve](../breakdown_curve_form_fit_2026_09_26.md)) would not remove a per-energy
  store, since none was found. Its case is smoothness and compute.

## Side-of-cut probe (2026-09-26)

Question: would representing each cut by its lighter side (the neutral loss of a large fragment)
remove the whole-molecule representation from the token? A `side` scope in the probe rewrites one
side's pool per candidate. The heavier side is the fragment when its `c_mass` precursor-mass
fraction is at least 0.5, else the complement. `popmean` substitutes the batch mean of that
side's rows, which removes identity. `molmean` substitutes the molecule's mean, which keeps
identity and removes which-candidate detail. The precursor has no lighter side, so the `small`
modes leave it alone. Same checkpoint, sample and reads as above; 24 tests pass.

| condition | train drop | val drop | damage ratio | excess gap closed |
| --- | ---: | ---: | ---: | ---: |
| heavier side, `popmean` | 0.127 | 0.072 | 1.78 | 0.041 |
| heavier side, `molmean` | 0.055 | 0.034 | 1.59 | 0.013 |
| lighter side, `popmean` | 0.252 | 0.178 | 1.42 | 0.037 |
| lighter side, `molmean` | 0.199 | 0.145 | 1.37 | 0.024 |
| cc pools (reference) | 0.393-0.494 | 0.325-0.395 | 1.21-1.25 | — |

By NCE bin (damage ratio, val drop):

| condition | ≤ 20 | 20-40 | 40-60 | 60-100 | 100-200 |
| --- | --- | --- | --- | --- | --- |
| heavier side, `popmean` | 2.77, 0.026 | 1.88, 0.045 | 1.81, 0.070 | 1.62, 0.115 | 1.68, 0.099 |
| lighter side, `popmean` | 2.03, 0.034 | 1.60, 0.071 | 1.51, 0.113 | 1.51, 0.239 | 1.33, 0.411 |

- **The heavier side leans to memorization.** Its identity costs train 1.78x what it costs val
  overall and 2.8x at NCE ≤ 20, where val barely uses it (0.026). Val's drop does not depend on
  molecule novelty (Spearman 0.03).
- **The lighter side carries what transfers.** Val loses 2.5 times more without it, most at high
  NCE, where the lighter side is often the observed ion itself. Its val damage grows with
  similarity to train (0.133 to 0.286), the parts path above.
- **The lighter side still names the molecule.** At 1.42x it sits above the pools. Both pools are
  built from global-attention embeddings, so a neutral-loss-only token keeps identity unless
  that side is pooled from a local encoder.
- **Not the key.** The heavier side's excess gap closed (0.041) is a third of global
  attention's (0.12-0.13). A token without it removes one memorizing input, and the rest of the
  lookup can re-form through the boundary block and the candidate set.
- **What the heavier side gives val.** Per-candidate detail costs 0.034 (`molmean`), and identity
  a further 0.038. A coarse, molecule-level summary in its place keeps the first part.

## Allocator set-message probe (2026-09-26)

Question: once the token holds no whole-molecule input (arm `nl2`), the allocator's attention
over a spectrum's candidates is the remaining route for a molecule key. Is that route
memorizing?

A `setmsg` scope rewrites, in every allocator attention layer, the message each candidate
receives from its set: the `candidate_attention` output before the residual add.
- `popmean` substitutes the layer's batch-mean message, which removes all set information. It
  is recorded on the same batch's clean pass, because the head runs size buckets one after
  another.
- `molmean` substitutes the mean message of the candidate's own spectrum. It keeps the set
  summary and removes candidate-specific retrieval from the set.
- The existing `set_drop` (50% of each spectrum's candidates hidden from the keys) runs alongside.

Same checkpoint, sample and reads as above; 29 probe tests pass.

| condition | train drop | val drop | damage ratio | excess gap closed |
| --- | ---: | ---: | ---: | ---: |
| set message, `molmean` | 0.078 | 0.023 | **3.37** | 0.050 |
| set message, `popmean` | 0.113 | 0.040 | **2.82** | 0.065 |
| `set_drop` (50% of keys) | 0.038 | 0.022 | 1.75 | 0.012 |

Against the other routes on the same f0 batches:

| route | damage ratio | excess gap closed | val drop |
| --- | ---: | ---: | ---: |
| allocator set message, `molmean` | 3.37 | 0.050 | 0.023 |
| allocator set message, `popmean` | 2.82 | 0.065 | 0.040 |
| encoder global-attention message, `popmean` | 2.58 | 0.123 | 0.090 |
| heavier side of the cut, `popmean` | 1.78 | 0.041 | 0.072 |
| boundary block, `molmean` (route map) | 1.47 | 0.085 | 0.353 |
| CE removed (`ctx_popmean`) | 1.37 | 0.045 | 0.275 |
| cc pools (reference) | 1.21-1.25 | — | 0.33-0.40 |

By molecule novelty, val damage under `molmean` is −0.004 at nearest-train Tanimoto ≤ 0.4 and
0.040 above 0.8. Train's is flat (0.066-0.083). By NCE the ratio is 6.4x at ≤ 20 and 2.2-3.8x
above.

- **The set message acts as a lookup.** Train uses it on every molecule, and val only on close
  analogues. That is the same signature global attention showed (val damage 0.028 → 0.131 with
  similarity).
- **Candidate-specific retrieval is the most memorizing part.** Keeping each spectrum's set
  summary and removing only the per-candidate message costs train 3.4 times what it costs val.
  Removing the summary as well adds 0.035 train and 0.017 val (2.1x).
- **Hiding keys is the weaker lever.** With half the keys hidden, retrieval survives (1.75x,
  excess 0.012). Substituting the message goes to the route itself.
- **Val barely needs it.** Removing all set information costs val 0.040, a tenth of what the
  pools carry. A training-time substitution of the set message, `gm1`'s form moved into the
  allocator, targets a route that val barely uses and that `nl2` leaves open.

## Limits

- **One checkpoint, one seed, one task.** f0 only, unmerged `[M+H]+`/FT.
- **Head-level substitution.** Contexts at shifted NCE are on the encoder's manifold. Most ±1
  targets are NCE values absent from training, so ±1 cannot show how a smooth encoding would
  behave. The symmetric damage rules out a per-energy key only within this encoding.
- **The part definition is one choice.** It counts a bond environment's presence anywhere in a
  train molecule, not as an observed cut, and attributes peaks by 0.01 Da bin with the most
  familiar explanation. Isomeric explanations are not resolved.
- **The per-peak error is not the cosine.** It is a log ratio on sum-normalized intensities,
  weighted by target share.
- **Samples.** 2,500 molecules per split for parts; 60 batches per split for the probe.
