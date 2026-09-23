# MASShatter overfit read-out and cosine-ceiling decomposition, 2026-09-22

> **Status:** FINDING · **Updated:** 2026-09-22
> Read-only W&B review of the 25 runs created 2026-09-21 and 2026-09-22 on the merged
> MA/MI base. The train/validation gap is **large (0.151 cosine), monotone, and never
> plateaus** — but it is **not a coverage gap**: train and validation candidate coverage
> agree to 0.006 `wrecall` while the model's cosine differs by 0.151, so **~98% of the
> generalization gap is intensity allocation among peaks the model already proposes**.
> Three LR-decay arms are the first thing in this family to clear the replicate floor.
> A follow-up intensity-memorization audit records five plausible shortcut routes and an
> ordered measurement/intervention gate. No code, config or run was changed for this review.

## Scope and method

Read-only queries against `frag-gnn/frag-gnn` on 2026-09-22, branch
`research/masshatter-ion-h-selection` at `aff8f749`. Population: all `mass_shatter` runs
created on 2026-09-21 and 2026-09-22 (15 four-hour campaign arms, 2 `c4a` base replicates,
4 `loss_objective` arms, 3 `lr_decay` arms, the 630-epoch horizon probe pair), plus a
400-run scan for historical `opt_cos_sim` instrumentation. Nothing has been running since
2026-09-22 06:51 UTC.

Metric is `val_cos_sim_0.01_epoch/mean` throughout; `_best/mean` is named explicitly where
used. Per-arm trajectories were re-aggregated by `epoch` from run history, forward-filling
within each epoch, because train and validation auxiliary metrics are logged on alternating
steps — a naive `dropna` on the validation key silently drops every train-side row. This is
the same trap recorded in the [2026-09-21 campaign read-out](masshatter_campaign_readout_2026_09_21.md).

All comparisons are at matched epoch, never at matched wall clock.

Section 7 extends the W&B read-out with a static code/config audit and read-only measurements
of the processed `data/split/nist20_ma_mi_inchikey` split under the active
`config/massshatter/ma_mi_merged/nist20_ma_mi_merged_fast_batch512_cuda_lb64.yml`
configuration. Those follow-up measurements are labelled as proxies or exposure imbalances,
not causal attributions.

## Headline

1. **The overfit is real and unbounded in epochs.** Train − validation cosine reaches
   **0.151** by epoch 127 and is still rising at +0.003/16 epochs. After epoch 32, every
   0.054 of train-cosine improvement buys **0.011** of validation — a 5:1 ratio.
2. **It is not a coverage gap.** Train `wrecall` 0.8975 vs validation 0.8911 — a 0.0064
   difference — against a 0.1507 cosine difference. The candidate set is equally good on
   both sides; what does not transfer is *which* candidate gets the intensity.
3. **Validation primary loss and validation cosine disagree late in training.** The primary
   loss turns first in all six instrumented runs, but by **6 to 476 epochs**, not uniformly:
   the 630-epoch control leads by 476, while the CE control `L_ce_ctrl` leads by **6**, i.e.
   not at all. The 630-epoch control's primary loss ends **0.096 worse** at epoch 629 than at
   its epoch-79 optimum while its cosine is flat. Cosine does **not** fall (see §3).
4. **Whether shrinking the gap raises validation is still untested** — an earlier draft of
   this read-out claimed it does not, and that claim was unsupported. The four-objective
   ordering is algebraically forced (§2), and in the 8-arm regularization sweep the
   gap/validation correlation is +0.72 but flips to **−0.67** when the single arm `drop03`
   is dropped (§2.1). The one arm that narrowed the gap without suppressing train cosine,
   `wd10`, moved validation **+0.0015** — the right sign, inside the ±0.007 screen
   resolution, at 10% of training exposure.
5. **Three LR-decay arms clear the floor.** `d3` (cosine decay + 1000-step warmup) reaches
   **0.7357** against a control mean of 0.7286 whose two replicates span 0.0007. First
   non-null in this family since the `tolerance_consensus` change.
6. **The 128-epoch screen length is confirmed.** Across the 15 arms that reached ~epoch 205,
   validation cosine gains a **median +0.0031** between epoch 128 and best — at the recorded
   replicate floor.

## 1. Instrumentation: train cosine is logged in 4 of 25 runs

`compute_train_auxiliary_metrics: True` in the four `loss_objective_2026_09_21` arms only.
It is `False` in the three `lr_decay` arms, both `c4a_base` replicates, the 630-epoch
horizon probe, and all 15 four-hour campaign arms. This is a statement about *these 25
runs*: the separate 8-arm regularization sweep of 2026-09-21 logged train cosine on every
arm, and it remains the only matched-epoch source of gap data in this model family.

**`c4a_base_s1` and `c4a_base_s2` carry the `train_aux_metrics` W&B tag while their resolved
config says `False`.** The tag is not evidence; read the config key.

Consequence: the arms one would actually want the overfit answer for — the LR schedules and
the feature-bundle attribution arms — cannot answer it. "Did cosine decay narrow the gap, or
relocate the same overfit to a better point?" is currently unmeasured.

`opt_cos_sim` is likewise off in every 2026-09 run (`auxiliary_scores` is
`['cos_sim', 'recall', 'precision', 'wrecall', 'wprecision']`). See §4.1.

## 2. The overfit: magnitude and shape

`L_ce_ctrl` (`wttzk2ot`, cross-entropy control, batch 256x2, `lb32`):

| epoch | 8 | 16 | 32 | 48 | 64 | 80 | 96 | 112 | 127 |
|---|---|---|---|---|---|---|---|---|---|
| train cosine | 0.7079 | 0.7688 | 0.8261 | 0.8463 | 0.8538 | 0.8667 | 0.8720 | 0.8760 | 0.8805 |
| val cosine | 0.6693 | 0.7019 | 0.7189 | 0.7226 | 0.7227 | 0.7272 | 0.7290 | 0.7280 | 0.7298 |
| **gap** | 0.0386 | 0.0668 | 0.1072 | 0.1237 | 0.1312 | 0.1395 | 0.1430 | 0.1480 | **0.1507** |
| train primary loss | −2.672 | −2.914 | −3.130 | −3.213 | −3.246 | −3.297 | −3.320 | −3.339 | −3.358 |
| val primary loss | −2.523 | −2.672 | −2.737 | −2.739 | −2.717 | −2.730 | −2.739 | −2.735 | −2.726 |

The gap decelerates but never plateaus. Final gaps across the four instrumented arms:

| arm | run | train cos | val cos | **gap** |
|---|---|---|---|---|
| cross-entropy (control) | `wttzk2ot` | 0.8805 | 0.7298 | **0.1507** |
| conditional cross-entropy | `agzpan55` (crashed ep115) | 0.8718 | 0.7250 | 0.1468 |
| Hellinger, closest matching | `oykkffls` | 0.8646 | 0.7226 | 0.1420 |
| cosine, closest matching | `0fku99sf` | 0.8380 | 0.7043 | 0.1337 |

The only configuration difference among the last three and the control is `loss_type`.

Note what this table cannot show. Train cosine is rank-identical to validation cosine across
the four arms and spans 1.7x more (0.0425 vs 0.0255); regressing val on train gives a slope of
**0.605**. Because that slope is below 1, gap = train - val is a strictly increasing function
of train, and train tracks val — so **gap and validation have to rank together here**. The
agreement is arithmetic, not evidence about overfitting. Read plainly these four rows say only
that `cosine_distance` fits worse on both sides.

### 2.1 The gap-versus-validation relationship rests on one arm

The 8-arm [regularization sweep](masshatter_campaign_readout_2026_09_21.md) is the only
matched-epoch test with train cosine on every arm, so it is the one place the question can
actually be asked:

| statistic | value |
|---|---|
| corr(gap, val) over all 8 arms | **+0.723** |
| corr(gap, val) excluding `drop03` | **−0.673** |
| corr(train, val) over all 8 arms | +0.914 |

A correlation that reverses sign on the removal of one point is that point. The arms are
ordered by overall fit quality (r = +0.914 train-to-val), and `drop03` narrows the gap by
suppressing train cosine (0.8139 vs 0.8449 control mean) rather than by lifting validation.
Set it aside and the surviving trend runs the *other* way: smaller gap, better validation.

`wd10` is the case that matters and it was never run to completion. It narrowed the gap
0.007 and moved validation **+0.0015** — correct sign, but the screen resolution is ±0.007
and every arm died at epoch 64 of 630 with validation still climbing. The sweep's own
caveat applies: that screen is structurally biased against arms trading early fitting speed
for late generalization.

## 3. Validation primary loss and validation cosine disagree, by a variable margin

All losses below are `val_primary_loss_epoch/mean` — the spectrum term, not the total.
The total also carries a proposal term that degrades *more* (+0.06 to +0.14) and turns even
earlier (epochs 26-47), so reading the total overstates the spectrum effect. Absolute values
are not comparable across runs (differing `oos_loss`); only each run's delta to its own
optimum is.

| run | best primary-loss epoch | best val-cosine epoch | lead | primary loss at last epoch | vs its own optimum |
|---|---|---|---|---|---|
| `c4a_base_s1` | **52** | 122 | 70 | −2.7182 | +0.038 |
| `d3_cosine_warmup1000` | **78** | 121 | 43 | −2.7465 | +0.018 |
| `long_ctrl_630` (`tkkesus9`) | **79** | 555 | **476** | −2.6583 | **+0.096** |
| `a1_prenorm` | **120** | 182 | 62 | −2.7474 | +0.031 |
| `L_ce_ctrl` | 116 | 122 | **6** | −2.7255 | +0.038 |
| `L_conditional_ce` | **53** | 100 | 47 | −2.9133 | +0.032 |

The 630-epoch control is the clearest case, with two corrections to an earlier draft of
this section. Its primary loss peaks at epoch 79 and is **0.096 worse** by epoch 629 — but
**not monotonically**: only 270 of the 550 post-peak epochs move worse, so this is a noisy
upward drift, not a clean turn. And validation cosine does **not** fall underneath it. From
epoch 79 to 629 cosine moves **+0.0042** (0.7267 → 0.7309, never below 0.7214). The curve is
flat because it is flat, not because the checkpoint selector is holding it up; the selector
chooses which checkpoint ships, it does not shape the curve.

What the disagreement does show is memorization in the *calibration* sense — the model grows
more confidently wrong on validation while its rank ordering, which is what cosine sees,
stops changing. That is consistent with §4's conclusion that the defect is comparative
ranking, and it is not by itself a reason to believe late training damages cosine.

This extends, and does not restate, the
[horizon probe](masshatter_campaign_readout_2026_09_21.md) result that validation cosine
saturates by ~epoch 100.

## 4. Why validation cosine will not move: the decomposition

### 4.1 It is not coverage

Coverage metrics on the same four instrumented runs, at epoch 127:

| metric | train | val | Δ |
|---|---|---|---|
| `wrecall` — true intensity matched by a predicted peak | 0.8975 | 0.8911 | **0.0064** |
| `recall` — true peak count matched | 0.6923 | 0.6814 | 0.0109 |
| `wprecision` — predicted intensity landing on a real peak | 0.9260 | 0.8755 | 0.0505 |
| `precision` — predicted peak count matching | 0.0651 | 0.0628 | 0.0023 |
| **`cos_sim_0.01`** | **0.8805** | **0.7298** | **0.1507** |

The three loss objectives agree to within 0.002 on every coverage number
(`wrecall` train 0.8954-0.8975, val 0.8906-0.8911), so this is a property of the candidate
set, not of the objective.

**The candidate set covers validation molecules as well as it covers training molecules.**
Whatever the model memorizes, it is not "which fragments exist" — it is "which of them
carries the intensity".

**Caveat on the train-side numbers.** Train auxiliary metrics are computed inside
`training_step`, so they are measured in training mode: BatchNorm uses batch statistics and
dropout is active. The [regularization audit](masshatter_regularization_audit_2026_09_19.md)
measured that BatchNorm alone makes the candidate set optimized in training diverge from the
one evaluated (Jaccard 0.317 under `batch`, exactly 1.000 under `layer`/`graph`/`none`), so
"train cosine 0.8805" is a train-mode number, not a clean train-set evaluation. This does not
undermine the decomposition: if the train-mode candidate set were materially better, train
`wrecall` would exceed validation `wrecall` by more than 0.0064, and it does not. It does mean
the 0.1507 figure is an upper bound on the true train/validation cosine difference. Settling
it exactly requires an eval-mode pass over the training split, which no run performs.

Converting the L1 `wrecall` difference into a cosine ceiling: three 2026-08-26 MASShatter
runs logged both `wrecall` and `opt_cos_sim` (the oracle that keeps true intensity on every
matched bin and zeroes the rest), giving a local slope of **0.41 cosine points per `wrecall`
point** — (0.9111, 0.9577), (0.9084, 0.9565), (0.8746, 0.9426). At that slope the 0.0064
`wrecall` difference is worth **~0.0027** of achievable cosine, i.e. **under 2% of the 0.151
gap**. The remaining ~98% is allocation.

The same conclusion falls out of FraGNNet on this exact split
(`data/split/nist20_ma_mi_inchikey`, `a5goxclw`, 2026-08-19), which did log the oracle on
both sides: **train `opt_cos_sim` 0.9271, validation 0.9243** — a 0.003 difference — while
the model sat at 0.7043 validation.

**Open measurement.** No 2026-09 run logs `opt_cos_sim`, so the *current* merged MA/MI
support ceiling is an extrapolation (~0.95 from the slope above), not a measurement. Adding
`opt_cos_sim` to `auxiliary_scores` on the next control costs one metric and settles it. It
is not the same quantity as the 0.937/0.959 `cos_hun` figures in
[the DAG oracle ceiling note](../fragnnet/dag_oracle_ceiling_nist20.md) — that is a model-free DAG
coverage bound under Hungarian matching on the test split; `opt_cos_sim` is a per-run bound
on the support the model actually emits.

### 4.2 It is not capacity, and no objective, regularization or schedule arm has reached it

Twenty-five arms since 2026-09-19 span **0.7043 to 0.7364** on validation cosine, and
everything except `cosine_distance` and two known-bad arms sits in **0.722-0.736**:

| family | arms | best | worst | span |
|---|---|---|---|---|
| loss objective (CE / cond-CE / Hellinger / cosine) | 4 | 0.7322 | 0.7043 | 0.0279 |
| regularization + normalization (§1 of the 09-21 sweep) | 6 | — | — | all inside ±0.0025 of control |
| attention / residual architecture (`a1`-`a7`, `b2`, `b3`) | 6 | 0.7364 | 0.7264 | 0.0100 |
| LR schedule (`c4b_*`, `d1`-`d3`) | 7 | 0.7357 | 0.7282 | 0.0075 |
| proposal warmup (`c4c_*`) | 3 | 0.7319 | 0.7218 | 0.0101 |
| dropout (`c4d_*`) | 2 | 0.7313 | 0.7282 | 0.0031 |

Meanwhile train cosine reaches 0.88. Capacity is not the constraint — the model can fit
these spectra; it cannot transfer the fit. The
[unmerged generalization audit](masshatter_unmerged_overfitting_2026_09_16.md) found the same
shape on the other track (a growing loss gap in mature runs), but it explicitly declines to
call that plateau an architecture-invariant ceiling. That caution applies here: 25 arms
inside 0.01 is strong evidence that these *knobs* are exhausted, not proof that no
architecture can do better.

### 4.3 What is actually missing: a transferable sibling-ranking rule

The split is by 2D InChIKey, so validation molecules are structurally novel. The model must
decide, for ~241 mass-feasible candidate components per spectrum, which ~22 carry intensity
and in what proportion. It concentrates well (validation `wprecision` 0.876 — the intensity
is not diffuse) and it finds the right peaks as often as the candidate set allows
(`wrecall` 0.891 vs a ~0.95 ceiling). What it gets wrong is the *relative* ordering.

Several independent probes already point at the same defect and are worth reading together
rather than re-running:

- [Dominant-peak outcome decomposition](../fragnnet/dominant_peak_outcome_decomposition.md) — base-peak
  failure is 7:1 ranking versus DAG coverage; misses are one neutral loss away.
- [Fragment intensity ranking findings](../fragnnet/fragment_intensity_ranking_findings.md) — the error
  is rank/identity, not calibration (status: provisional nodeMLP evidence).
- [cc-corruption probe](../fragnnet/cc_corruption_probe_finding.md) — fragment identity *is* represented
  and used; the gap is comparative scoring, not features.
- [GLACIER gap CE decomposition](../fragnnet/glacier_gap_ce_decomposition_2026_09_21.md) — the remaining
  gap is concentrated at high collision energy, where branching decisions multiply.

None of the 25 arms above changes the comparative-scoring mechanism. They change how hard,
how smoothly, or how long the same mechanism is fit. That is why they all land within 0.01.

### 4.4 The 0.73-vs-0.80 difference is task scope, not a regression

Worth pinning, because the numbers sit side by side on the same W&B project. The recorded
0.8010 (`8v0i0fq7`) and the current 0.7357 are **different tasks**:

| | `8v0i0fq7` (0.8010) | current `ma_mi_merged` (0.7357) |
|---|---|---|
| split | `data/split/nist20_inchikey` | `data/split/nist20_ma_mi_inchikey` |
| adducts | `[M+H]+` | `[M+H]+`, `[M-H]-`, `[M+Na]+`, `[M+Cl]-` |
| instruments | FT | FT, QTOF |
| fragmentation modes | — | HCD, CID |

Both use `merge_method: tolerance_consensus`, so the 2026-09-03 target change is not the
confound here. Given the recorded per-mode gap (CID 0.547 vs HCD 0.781, QTOF ~0.616), adding
QTOF and CID mechanically lowers the mean. Do not read 0.73 as a loss of 0.07 against 0.80.

**Scope is not the only difference.** `8v0i0fq7` also runs
`spec_params.filter_subformula_peaks: True`, and **every run of the current campaign has it
`False`**. Cross-tabulating 256 scored `mass_shatter` runs by split, scope and filter:

| split | adducts | instruments | `filter_subformula_peaks` | n | best | median |
|---|---|---|---|---|---|---|
| `nist20_inchikey` | 1 | FT | **True** | 19 | **0.8010** | 0.7941 |
| `nist20_inchikey` | 1 | FT | False | 150 | 0.7949 | 0.7255 |
| `nist20_ma_mi_inchikey` | 4 | FT+QTOF | False | 58 | 0.7364 | 0.7183 |
| `nist20_ma_mi_inchikey` | 4 | FT+QTOF+IT | False | 23 | 0.6933 | 0.6442 |

On matched split and scope the filter is worth **+0.0061 best-vs-best** (0.8010 vs 0.7949);
the much larger median difference is a population artifact, since the 19 filtered runs are all
recent and mature while the 150 unfiltered ones include the whole architectural history. So
scope carries roughly 0.06 of the difference and the filter roughly 0.006 — but the filter's
share is **not a like-for-like model comparison**: `clean_spectra` runs at dataset load
(`base_dataset.py:430`) and each arm is scored against its own cleaned target, so part of that
+0.0061 is the removal of peaks the model was failing on. The same mechanism produces a uniform
+0.026 `val_candidate_wrecall` across filtered/unfiltered pairs, which is a denominator artifact
rather than learning.

**The filter has never been run on this base: 0 of 58 `nist20_ma_mi_inchikey` runs enable it.**
The adduct damage that made it unsafe for `[M+Na]+`/`[M+Cl]-` was fixed on 2026-09-15 (CMF
carriers are now always enumerated), so the four-adduct scope is no longer a blocker.

## 5. The one thing that cleared the floor: LR decay

Every run before 2026-09-21 trained at constant `lr: 1e-3`, because
`SpectrumPL.configure_optimizers` dropped `lr_schedule_type` (fixed 2026-09-21; see the
[regularization audit](masshatter_regularization_audit_2026_09_19.md)). These are the first
arms to actually get a schedule.

| arm | run | best val cosine | best epoch | vs control mean |
|---|---|---|---|---|
| `c4a_base_s1` (seed 397623) | `5qws8g9x` | 0.7282 | 122 | — |
| `c4a_base_s2` (seed 811237) | `6plumx6r` | 0.7289 | 123 | — |
| **control mean** | | **0.7286** | | **replicate spread 0.0007** |
| `d1` cosine, `lr_min_ratio` 0.01, no warmup | `oqobubu7` | 0.7329 | 104 | **+0.0043** |
| `d2` cosine, `lr_min_ratio` 0.10, no warmup | `2369y1j0` | 0.7326 | 122 | **+0.0040** |
| `d3` cosine, `lr_min_ratio` 0.01, warmup 1000 | `rcfzf67p` | **0.7357** | 121 | **+0.0072** |

The resolved-config difference between `d1` and `c4a_base_s1` is five LR keys and nothing
else, at the same seed; `d1` and `d3` differ only in `lr_warmup_steps`. All three exceed the
0.003-0.005 replicate floor recorded for this family, and `d3` exceeds it by 1.4-2.4x.

**Caveat: n=1 per arm.** The control is n=2 and unusually tight (0.0007), which is narrower
than this family's historical floor, so the margin may be flattered. Replicate `d3` before
promoting it.

Note also that `d3` has the *widest* train/validation loss gap of the 128-epoch set
(−0.656 at epoch 127, against the control's −0.632) while having the best validation cosine.
Another instance of §4's point: the gap is not the thing to minimize.

## 6. Horizon: the 128-epoch screen is the right length

Fifteen arms reached ~epoch 205 before their four-hour wall-clock kill. Validation cosine
from epoch 128 to each arm's own best:

median **+0.0031**, mean +0.0035, max +0.0074 (`b2_identity_residual`, which is the worst
arm overall and still climbing from a low base), over ~77 additional epochs.

Against the 0.003-0.005 replicate floor, **the extra 77 epochs buy nothing resolvable**.
The 2026-09-21 recommendation to screen at ~128 epochs rather than 630 stands, now
confirmed on 15 arms rather than one probe.

## 7. Intensity-memorization threat model

The actionable objective is narrower than preventing every possible shortcut: prevent the
model from memorizing molecule-specific intensity allocations while retaining transferable
fragment identity. The evidence above establishes the *location* of the problem — intensity
allocation after candidate construction — but it does not yet identify a single causal
mechanism. The current audit leaves five credible routes:

| route | current evidence | status |
|---|---|---|
| Global candidate-set fingerprint | The intensity transformer jointly attends over all selected candidates, so candidate identities and their configuration can identify a molecule even without an explicit molecule ID. The allocator is sufficiently expressive to reach 0.8805 train cosine while stopping at 0.7298 on validation. | **Plausible, unisolated.** Requires a smaller-head or candidate-subset intervention. |
| Unequal repeated exposure by molecule | On the active merged training split, 15,514 molecules produce 45,639 conditioning groups. Groups per molecule range from 1 to 10; the 37.3% of molecules with at least four groups contribute 58.0% of all groups. The active configuration uses neither molecule-balanced sampling nor loss weighting. | **Measured exposure imbalance; causal effect untested.** |
| Conditioning shortcuts | QTOF rows usually lack raw NCE, so the dataset derives it from ACE, precursor m/z and charge. Among 5,254 validation QTOF rows, only 5 carry raw NCE; derived NCE correlates −0.517 with precursor m/z while raw ACE correlates +0.015. The NCE embedding can therefore act partly as a precursor-mass feature. | **Measured proxy; causal effect untested.** |
| Population and precursor priors | On the exact merged validation target, a precursor-only predictor reaches mean cosine 0.275 (median 0.178; p90 0.754), and the [dominant-peak decomposition](../fragnnet/dominant_peak_outcome_decomposition.md) shows the model over-calls the precursor at low CE. These priors are useful on average but can replace fragment-specific ranking. | **Measured shortcut behavior; contribution to the 0.151 gap unresolved.** |
| Non-causal fragment proxies | MASShatter exposes cut count and component size while mean pooling loses relational cut topology; [the cut-set audit](../../plans/masshatter/masshatter_cut_set_features_plan.md) finds 96.8% of fragments have at least two cuts and 55.4% have two cuts sharing a ring. FraGNNet's legacy `depth` is a proxy for cut count on plain BFS DAGs and does not retain that meaning for SMARTS/multicut DAGs. | **Representation defect measured; memorization effect untested.** |

Two tempting explanations are already ruled out or materially weakened:

- **Direct proposal-score leakage is absent in the current lead.** With
  `proposal_anchored: true` and `proposal_selection_only: true`, the intensity path excludes
  proposal probabilities and uses the residual fragment logits alone. The *selected candidate
  set* remains visible and can still be a fingerprint, but removing the base probability feature
  again is not an experiment.
- **Merged targets are not an established cause.** The
  [unmerged audit](masshatter_unmerged_overfitting_2026_09_16.md) also found substantial
  overfitting. A matched merged/unmerged comparison could test the target construction, but
  unmerging is not currently a supported prevention strategy.

Do not remove or broadly corrupt fragment identity to address this. The
[cc-corruption probe](../fragnnet/cc_corruption_probe_finding.md) shows that identity is load-bearing for
FraGNNet. The intervention should remove opportunities to memorize allocation, not remove the
chemical information needed for transferable ranking.

### 7.1 Measurement gate

Before attributing improvement to memorization prevention, evaluate the same fixed checkpoint
in eval mode on both train and validation and report:

1. molecule-balanced as well as group-balanced cosine;
2. `wrecall` and `opt_cos_sim`, to keep support changes separate from allocation changes;
3. validation cosine for scaffolds seen versus unseen in training; and
4. train/validation cosine and calibration gaps at the same epoch.

The scaffold slice matters because 65.9% of validation molecules in the active InChIKey split
have a Bemis-Murcko scaffold represented in training. Aggregate validation is therefore not a
pure structure-OOD test. The existing train metric is also collected in training mode and is
group-weighted, so the headline 0.1507 gap remains an upper bound rather than the final
molecule-balanced memorization measurement.

### 7.2 Ordered interventions

Use matched 128-epoch arms and keep proposal coverage visible throughout:

1. **Replicate full-length `wd10`.** It is the only existing arm that narrowed the gap without
   suppressing train cosine and moved validation in the right direction, although the result is
   still inside screen noise.
2. **Replicate `d3` with train auxiliary metrics.** It has the best validation result, but the
   current run cannot say whether the schedule reduced memorization or simply optimized the same
   allocator better.
3. **Regularize only the intensity allocator.** Hold molecular and proposal dropout at 0.1;
   test intensity-head dropout around 0.2 and a genuinely smaller 128-dimensional, one- or
   two-layer candidate-attention head. The global `drop03` arm regularized three subsystems at
   once and underfit, so it does not answer this question.
4. **Balance training by molecule.** Test inverse-groups-per-molecule loss weighting or one
   randomly selected conditioning group per molecule per epoch. `train_sample_weight` currently
   changes metric aggregation, not the training loss, so merely setting that key is insufficient.
5. **Remove the QTOF mass proxy.** Represent ACE directly for QTOF, or carry ACE and NCE as
   distinct conditioning variables, in a matched arm.
6. **If the cheap arms fail, attack candidate-set fingerprinting directly.** Train two stochastic
   candidate-subset views of each spectrum and require consistent intensities on shared
   candidates. This is a new objective and should follow, not precede, the diagnostic arms above.

Early stopping at about 128 epochs limits wasted exposure but is not itself a solution: the
allocation gap is already 0.1507 at epoch 127. Likewise, adding cut-topology features may improve
causal ranking, but adding features alone also increases identification capacity; pair such an
arm with the measurement gate rather than assuming it prevents memorization.

## 8. What follows

Ordered by information per GPU-hour.

1. **Turn on the two metrics that make this answerable** on the next control:
   `compute_train_auxiliary_metrics: true` and `opt_cos_sim` in `auxiliary_scores`. Without
   them the LR and feature families cannot be read for generalization at all, and the
   support ceiling stays an extrapolation. Cost: eval-tensor overhead on train batches.
2. **Replicate `d3`** (n=2, 128 epochs) before promoting the LR schedule. This is the only
   live candidate in the family and it currently rests on n=1 against an unusually tight
   control.
3. **Deprioritize — do not close — regularization, objective and schedule screening.**
   Six regularization arms, four objectives, seven LR arms and two dropout arms have
   produced one result above the floor between them, and §4.1 argues they do not reach the
   defect. But §2.1 shows the gap lever itself was never cleanly tested: `wd10` narrowed
   the gap without suppressing train cosine and moved validation the right way, inside a
   ±0.007 screen at 10% exposure. One full-length `wd10` run would settle it; that is a
   cheaper open question than it looked. Schedule work remains worth doing for optimization
   speed, which is what `d3` appears to be buying.
4. **Attack comparative scoring instead.** The open, never-run item with a mechanism behind
   it is the marginal sibling-ranking target — a ranking loss on P(n) among siblings of a
   shared parent, which is the quantity §4.3's four probes all independently implicate.
5. **Do not extend training past ~128 epochs for screening.** The reason is cosine
   saturation, not damage: the 09-21 horizon probe shows the last 529 of 630 epochs buying
   +0.0017, inside the replicate floor. §3's primary-loss degradation is a calibration
   signal and does *not* show late training hurting cosine — an earlier draft of this
   recommendation implied it did.

## Run index

| label | run | state | last epoch | val cos best |
|---|---|---|---|---|
| `d1_cosine_min001` | `oqobubu7` | finished | 127 | 0.7329 |
| `d2_cosine_min010` | `2369y1j0` | finished | 127 | 0.7326 |
| `d3_cosine_warmup1000` | `rcfzf67p` | finished | 127 | 0.7357 |
| `ce_ctrl` | `wttzk2ot` | finished | 127 | 0.7322 |
| `hellinger_closest` | `oykkffls` | finished | 127 | 0.7247 |
| `cosine_closest` | `0fku99sf` | finished | 127 | 0.7043 |
| `conditional_ce` | `agzpan55` | crashed | 115 | 0.7255 |
| `hellinger_hungarian` | `vaaolmgc` | crashed | 31 | — (no usable data) |
| `c4a_base_s1` | `5qws8g9x` | finished | 127 | 0.7282 |
| `c4a_base_s2` | `6plumx6r` | finished | 127 | 0.7289 |
| `long_control_630` | `tkkesus9` | finished | 629 | 0.7318 |
| `warmup_e0_4_630` | `jvaj1lyw` | finished | 629 | — (see 09-21 read-out) |
| `a1_prenorm` | `q4t4udee` | crashed | 205 | 0.7364 |
| `a3_full_massbias` | `4fg0x5vw` | crashed | 180 | 0.7347 |
| `a6_ce_film` | `0gbdejd8` | crashed | 209 | 0.7364 |
| `a7_ce_both` | `zgm04inn` | crashed | 179 | 0.7341 |
| `b2_identity_residual` | `icii1zho` | crashed | 209 | 0.7264 |
| `b3_wd_groups` | `gjeqjp44` | crashed | 210 | 0.7305 |
| `c4b_lr_cos` | `vd2b0j7r` | crashed | 205 | 0.7341 |
| `c4b_lr_exp` | `ml4nge8s` | crashed | 209 | 0.7343 |
| `c4b_lr_cos_nowarm` | `a6xf1ve3` | crashed | 209 | 0.7315 |
| `c4b_lr_3e4` | `wqn9xaob` | crashed | 208 | 0.7282 |
| `c4c_pwarm_e2` | `1xl29zka` | crashed | 203 | 0.7319 |
| `c4c_pwarm_e4` | `7h1wx0ft` | crashed | 205 | 0.7318 |
| `c4c_pwarm_e15` | `c25ayosq` | crashed | 206 | 0.7218 |
| `c4d_drop02` | `zx7zxqrv` | crashed | 207 | 0.7282 |
| `c4d_drop_head03` | `kjkdr22o` | crashed | 208 | 0.7313 |

Most `crashed` states are four-hour wall-clock kills and are by design. `vaaolmgc` and
`agzpan55` are genuine early terminations, leaving the loss-objective screen at four usable
arms rather than six.
