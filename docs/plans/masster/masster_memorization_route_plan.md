# Closing MASSter's per-molecule allocation lookup on every route

> **Status:** ACTIVE · **Updated:** 2026-10-06
> Next check is the computed-energetics information gate (policy and protocol first); route-removal, architecture and conditioning screens are finished with no single-seed pass beyond `i3`/`i4`. Dated log: [history](../../history/masster_memorization_route_plan_log.md).

The memorization probe (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_memorization_route_probe_2026_09_23.md`)
found the lookup in the boundary block. But every candidate input MASSter builds comes from
the same globally attended atom embeddings, so blocking that one input can move the lookup
instead of removing it. This plan first maps every route by which the lookup can reach the
output, including two the probe cannot see: the encoder, and the ion/H head. It then removes
molecule identity at its source, or caps what passes after the token, and re-maps the routes
after each change. It complements the ranking-transfer (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_ranking_transfer_plan.md`)
arms `h1`, `h2` and `u2`. The identity-blind lability prior in the final score belongs to the
intensity-displacement plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_intensity_displacement_plan.md`) (its arm 2A). Adding
data is out of scope.


## Current summary

MASSter only, unmerged `[M+H]+`. Reads are on `i1` (`i50ig7rf`, `ORB`/`ORB_F` NIST20, seed 397623);
results on the old `FT` labels (`f0`/`f1`) are retired. Screen: tail mean of
`val_cos_sim_0.01_epoch/mean`, epochs 54-63; seed spread 0.0042; single-seed gate +0.005. Dated log:
[history](../../history/masster_memorization_route_plan_log.md).

**What was tried, with verdicts**

- **Instrument input (`i1`).** +0.0278 over `n4`; the eval-mode train/val gap narrows 0.183 -> 0.158
  while train fit rises, about 86% of the gap remains ([read](../../findings/masster/masster_i1_memorization_read_2026_09_28.md)).
- **Regularizers on `i1`.** `i3` (`k1` dropout + weight EMA 0.999, `xypkkfti`) +0.0102 passes; `i4`
  (`i3` + intensity pre-norm, `rd9rsdqx`) 0.7729, +0.0019 over `i3`, the best tail on the lineage. The
  cosine schedule `s1` passes against `c0` (+0.0090) but adds nothing on EMA. `w1` weight decay +0.0010.
- **Architecture on `i1`.** Full candidate attention `fa1` (`0vn8m2gr`) +0.0044, just under the gate;
  its seed replicate `fa1_s2` is +0.0026. `mz1` (m/z position) -0.0017, `mznl1` -0.0068 and `nlpe1`
  -0.0152 against `fa1`: mass as a token position is closed. Restricting attention costs cosine
  (`mw30` -0.0062, `mw10` -0.0117, `mlp1` -0.0112); `cg1` ends +0.0007 over `i1`, -0.0036 against `fa1`.
- **Route removal and noise (rerun on `i1`).** `r2` -0.0007, `r1` -0.0064, `r0` -0.0151, `nl1` -0.0068,
  `nl2` -0.0334, `sd1` -0.0016, `ce1`/`ce2` (NCE into the encoder) +0.0003/+0.0005; JEPA null
  (closed). Earlier `c0`-lineage arms: `k1` +0.0074, EMA `u2` +0.0094, `gm1` +0.0051 (one seed),
  `gm2`-`gm6`, `m2`, `sm1`, `sd1`, `am1`/`de1` null or single-seed.
- **Information gates on `i1`, all negative.** Post-cut descriptors <= +0.001, fragment connectivity
  `fc1` +0.0050 (below its 0.0076 control spread), structural alignment, reference information,
  cut-score probe and explicit peak-ranking pilot (all select zero), within-peak isomer split (no arm).
- **Floor.** 330 spectra / 119 molecules below 0.05 at instrument tolerance (train sample: 4); 88% are
  ranking failures with the candidates present, and no tested arm moves a floor chosen by an
  independent checkpoint ([read](../../findings/masster/masster_i1_worst_predictions_2026_09_30.md)).

**Open**

- Computed-energetics information gate ([design below](#computed-product-ion-energetics-information-gate-2026-10-01-proposed)):
  formal floor coverage 97%, but only 14 / 24 harder channels keep a distance-preserved product pair.
  Next: a bounded target-independent state/calculation policy and a train/calibration/confirmation
  protocol before fitting ([feasibility finding](../../findings/masster/masster_energetics_feasibility_2026_10_02.md)).
- Coverage, parallel: `so1` (templates-on sulfonamide O transfer) is prepared, not submitted.
  Cross-molecule episodic training is [deferred](#transfer-oriented-training-for-covered-ranking-failures-2026-10-01-deferred); ensembling stays last.

**Standing decisions**

- User: many routes in parallel, never a single-input block; `i1` is the control for new arms; rank
  worst predictions by `cos_hun_inst_tol`; adding data is out of scope. Constant LR everywhere
  ([decision C](../../notes/memorization_literature_2026_09_24.md)).
- A pass needs all five gates below against a same-code control at matched epoch; a narrower gap from
  lower train fit is underfitting, not a pass.

## Objective and acceptance

**Review, 2026-09-27:** the plan review (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_memorization_plan_review_2026_09_27.md`)
recommends separating held-out performance promotion from the mechanistic criteria below.
Route corruption measures dependence, not a guarantee that removing identity improves transfer;
the numerical seed-noise thresholds are screening heuristics. Historical gates and results are
preserved here. The review gives the proposed next measurement sequence; no new runs were launched.
The [further literature review](../../notes/masster_generalization_methods_2026_09_27.md)
adds concrete consistency, logit-regularization and optimizer experiments, with controls and
stop gates. These remain proposals and do not replace the running-arm queue.

Make the intensity allocation transfer to new molecules. On unmerged `[M+H]+`/FT:

- The eval-mode train/val cosine gap is 0.19, and coverage differs by 0.007.
- The ranking gap is 0.011 on a train sample and 0.121 on val.
- The base-peak hit rate is 84.5% on train and 56.7% on val
  (ranking decomposition (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_ranking_decomposition_2026_09_23.md`)).

The model fits each training molecule and ranks new ones poorly.

An arm passes only if all five hold against a same-code control at matched epoch:

1. **Validation.** The tail mean of `val_cos_sim_0.01_epoch/mean` over epochs 54-63 is at least
   +0.003 above the control. `3bej6fq7` sits at 0.7298 with an epoch-to-epoch change of 0.0013.
   No seed replicate exists, so a pass is replicated before promotion.
2. **Ranking.** The val ranking gap falls. The train ranking gap may rise.
3. **No rerouting.** The Step 0 route map, re-run on the arm's checkpoint, shows the total
   eval-mode gap falling. No route's train/val asymmetry at matched validation damage may grow
   beyond its control value.
4. **Novelty.** The gain is largest in the nearest-train Tanimoto bins at or below 0.6
   (variance-novelty (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_variance_novelty_2026_09_24.md`)).
5. **Coverage.** `wrecall` stays within 0.005 of the control.

Narrowing the gap by lowering train fit is underfitting, not a pass. `drop03` (−0.0149) did
that, with train cosine logged. `b3_wd_groups` and `e2` logged no train cosine, and narrowed only
the train-mode loss gap; `b3`'s change is inside that statistic's own floor.

## How MASSter builds an allocation, and where a lookup can sit

The forward pass, for the `f0` control, has six stages:

1. **Encoder.** GINE_GSA runs 2 layers at width 384 with BatchNorm and dropout 0.1. Each layer
   adds exact per-graph self-attention to the local message: `x = local_out + gsa_out`
   ([`nn_blocks.py:1289-1313`](../../../src/fragnnet/model/nn_blocks.py#L1289-L1313)). With
   `context_in_gnn: False`, the encoder never sees CE.
2. **Proposal.** The bond head reads those embeddings detached. It selects the forced
   precursor and single-cut masks, then masks in descending joint bond probability, stopping
   at 256 unique fragments.
3. **Token.** Each candidate's token has 1,285 columns
   ([`masster_model.py:2869-2972`](../../../src/fragnnet/model/masster_model.py#L2890-L2993)):
   - a fragment pool and a complement pool (mean atom embeddings, 384 each);
   - the boundary block (384): `CutBondEncoder([in, out, in⊙out])` on each cut bond's two
     endpoint embeddings, mean-pooled over the candidate's cut bonds;
   - the context vector (128), which is CE only on this task;
   - five descriptors: `c_cut_count`, `c_size`, `c_mass` (two columns) and `c_charge`.
4. **Allocator.** A linear projection 1,285 → 256 (SiLU, LayerNorm, no dropout) feeds 2 induced
   attention layers over the molecule's candidates (64 inducing points, dropout 0.1). The head
   has 2.46M parameters.
5. **Outputs.** A residual head gives one logit per fragment, softmaxed over the 256 fragments.
   An ion/H head gives 9 logits per fragment (one ion mode, H shifts −4 to +4), softmaxed within
   the fragment. A peak's log-probability is the sum of the two
   ([`masster_model.py:3276-3282`](../../../src/fragnnet/model/masster_model.py#L3297-L3303)).
6. **Loss.** Cross-entropy on matched target intensity. Each molecule has 12.4 train spectra on
   average, at different NCE.

Where the lookup can get its key (K), where it can be stored (S), and what drives it (I):

| # | route | MASSter evidence | measured? |
| --- | --- | --- | --- |
| K1 | **Global attention in the encoder.** Every atom embedding carries a whole-molecule read-out, so every token input derived from atoms identifies the molecule. The encoder is CE-blind, so a molecule's key is identical across its 12.4 spectra: about 800 presentations of the same key over 64 epochs. | No MASSter run compares GINE with GINE_GSA on the current lineage. The August GINE runs (val about 0.60) predate most of today's model. `g1` (5-layer encoder, no bond head, no `CutBondEncoder`) was −0.016. | no |
| K2 | **The boundary block.** Given the precursor, the oriented cut set fixes the fragment, and `in⊙out` multiplies two global read-outs. So this block separates the 256 candidates sharply *and* names the molecule. The encoder is 0.79M parameters with no dropout ([`base_model.py:857`](../../../src/fragnnet/model/base_model.py#L948)), and `candidate_atom_dropout` skips it by design. | Probe: at matched validation damage, boundary corruption leaves a 0.022 gap against 0.059-0.086 for the pools. `c4a_no_bbe` (endpoint mean instead of the encoder) was −0.0024, with no train metrics and no checkpoint. | yes |
| K3 | **Fragment and complement pools.** They name the molecule but separate candidates only weakly, since overlapping fragments share most of their atoms. | Probe: used symmetrically by both splits today. `e2`/`e3` (atom dropout on the pools only) were null. | yes |
| K4 | **Exact descriptors and inputs.** `c_mass` (log1p neutral mass plus precursor mass fraction) nearly fixes the formula. `c_charge` and the `a_charge` atom input are continuous Gasteiger values. | Probe: descriptors used symmetrically. Removing `c_mass` +0.0012, `c_charge` −0.0012, `a_charge` −0.0025, all inside the merged ±0.005 gate (campaign read-out §8 (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_campaign_readout_2026_09_21.md`)). | partly |
| K5 | **The candidate set.** In evaluation, the 256 selected fragments are a fixed function of the molecule, and the allocator attends over all of them. | Probe: `set_drop` closes 9-10% of the gap. Training already perturbs the set through BatchNorm, and the lookup forms anyway. (A random-weight probe measured train-versus-eval Jaccard 0.60-0.77 at dropout 0.1; the trained magnitude is unmeasured, regularization audit §2.2 (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_regularization_audit_2026_09_19.md`).) | yes |
| S1 | **Allocator capacity.** No dropout on the token projection or the cut-bond encoder. | `a2_full_attention` cut the head 43.6% and scored +0.0030, so val does not need this capacity. `h1`/`h2` are prepared. | partly |
| S2 | **The ion/H head: a second per-candidate store.** A 9-way H-shift distribution per fragment, read from the same contextualized token. | The probe corrupts the token before both heads, so it cannot say which head holds the memorized allocation. | no |
| I1 | **Exposure and objective.** Each key is seen about 800 times, and cross-entropy is minimized by fitting each molecule. | Train `cos_hun` 0.961 against val 0.762. Exposure is flat (Spearman 0.058): every molecule is memorized about equally. The precursor fallback appears only on unseen molecules. | yes |

Two more gaps are in the measurement itself. Runs log no train metrics. And the merged lineage
uploads no checkpoints, so `c4a_no_bbe` can never be probed. `t1`/`u1` (boundary-block dropout)
were dropped on the rerouting argument without a measurement.

**Why the trained model keys on the boundary block (hypothesis, from the wiring).** A lookup key
must name both the molecule and the candidate. The pools name the molecule but blur candidates
that share atoms. The descriptors separate candidates but do not name the molecule. The boundary
block does both. If it goes, the same key can be assembled from pool differences plus exact
descriptors, which is weaker but within the allocator's capacity. So the fix is either to strip
identity where it enters (K1), or to cap what the token can carry.

## Step 0: map every route, inference only

Run locally on `6h67zps9` and `3bej6fq7`. Each probe run takes about 45 min per checkpoint.

1. **Encoder scope (K1).** Add per-layer substitutions of `gsa_out` to
   `scripts/debug/masster_memorization/debug_pooled_embedding_memorization.py`:
   - `popmean`, the batch-population mean, which removes identity;
   - `xmol`, another molecule's mean, which swaps identity;
   - `zero`.

   Feed the corrupted embeddings to the intensity path only, so the candidate set stays fixed as
   in the existing scopes.
2. **Output scope (S2).** Corrupt the contextualized token before the residual head only, then
   before the ion/H head only. This says which output stores the memorized allocation.
3. **Descriptor and input scopes (K4).** Corrupt `c_mass`, `c_size`+`c_cut_count` and `c_charge`
   separately, and round `a_charge` to 0.1.
4. **Cross-molecule collision census.** The
   cut-descriptor finding (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_candidate_cut_descriptors_2026_09_22.md`)
   counts buckets within a molecule. A lookup needs buckets across molecules.
   - On MASSter's own selected candidates for about 2,000 train molecules, count the distinct
     training molecules per bucket for cut-bond environment multisets at radius 0, 1 and 2, and
     for `c_mass` binned at 1 Da and at 0.01 Da.
   - Report the share of candidates, and of target intensity, in buckets shared by at least 10
     training molecules. This sets the radius for a tied cut term (Step 2a).
   - Hash the environments the way the displacement plan's Phase 1.3 does, so the two plans
     share one census.

Read every scope at matched validation damage and as share kept above the floor, as the probe
finding does. Raw gap closed is misleading once both splits sit near the floor.

**Built 2026-09-24 (items 1-3; the census, item 4, is not built).** The probe gained these
scopes; the default scope list is unchanged:

- `gsa`, with modes `popmean`, `molmean` and `xmol`. It rewrites the global-attention message of
  both GINE_GSA layers on a second encoder pass, using forward hooks on `gsa_norms`, and swaps
  the rewritten atom embeddings into the component pool and the boundary block only. The
  proposal and the candidate set stay exact. `zero` was dropped as off-manifold. The unit test
  checks that proposal logits and candidate m/z are bit-identical under the rewrite.
- `acharge_round`: `a_charge` rounded to 0.1, through the same second pass.
- `ionh`, with modes `molmean` and `popmean`, on the head's ion/H logits. This replaces item
  2's token-level output scope: a corrupted token before the linear residual head only destroys
  the whole allocation on both splits, which carries no information.
- `feat_mass`, `feat_size` and `feat_charge`, each corrupting only its descriptor columns.

Tests are in `tests/test_debug_pooled_embedding_memorization.py` (16 pass). The run on both
production checkpoints uses 17 paired conditions and 60 batches of 512 per split:
`output/memorization_route_step0_2026_09_24/run_probe.sh`, read with `analyze_route_map.py`
beside it.

**Result, 2026-09-24 (route map (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_route_map_2026_09_24.md`)).**
Both checkpoints agree to about 0.01.

- **Global attention is where the key is written.** Replacing each atom's global message with
  the batch mean costs train 2.6 times what it costs val (the pools cost 1.2 times). It cuts the
  gap from 0.18 to 0.04 for a val drop of only 0.08-0.09.
- **The boundary block is where it is read** (1.46x).
- **The ion/H head is mostly generalizing** (1.28-1.30x), exact mass is weak, and `a_charge`
  precision is null.

Consequences for the plan:

- `r2` has the strongest evidence of the Step 1 arms.
- `k1` is weaker: the key is written upstream of the head, and `h1`'s allocator dropout was
  rejected.
- Steps 2d (coarse mass) and 2e (tied ion/H) drop in priority.
- A new targeted candidate is global-message dropout: in training, replace the global message
  with the batch mean for a random share of molecules. Built per atom as `gm1` on 2026-09-25.

## Step 1: first arms

Unmerged `[M+H]+`/FT, 64 epochs, `upload_checkpoints: true`, so Step 0 can re-run on each
checkpoint. The configs are in `config/masster/mh_ft_memorization_route_2026_09_24/`.
`c0_control.yml` is `f0_mh_ft_control.yml` plus naming and `masster_candidate_coverage_metrics:
true` (monitoring only, so criterion 5 reads off W&B). Each arm is `c0` plus its own keys, which
the resolved-config diff confirms. `c0` is the control, submitted with the arms, because the code
has moved since `3bej6fq7`. Generator (run on Fir, generates only):
`scripts/manual_slurm/fir/submit_masster_mh_ft_memorization_route.sh [all|arms|c0|r0|r1|r2|k1|m1]`.

| arm | keys against `c0` | question |
| --- | --- | --- |
| `r0_no_boundary` | `masster_candidate_boundary_block: false`, and `masster_boundary_bond_encoder: false` (the model rejects the encoder without the block) | Does the lookup reroute? A diagnostic, not a fix. |
| `r1_local_cut` | `masster_boundary_local_layers: 1`: the `CutBondEncoder` reads a separate one-layer plain GINE over raw atom features, with no global attention and no context | Does the cut term transfer once it cannot name the molecule (K2)? |
| `r2_no_gsa` | `mol_gnn_type: GINE` | Is global attention the identity source for every route (K1)? The proposal reads the same encoder, so read candidate coverage first. |
| `k1_key_dropout` | `masster_boundary_encoder_dropout: 0.2` (the `CutBondEncoder` hidden layer) and `masster_intensity_input_dropout: 0.2` (the projected token, after the projection's LayerNorm, before candidate attention) | Does noise on the layers that form the key stop it forming (S1)? |
| `m1_sink` | `masster_memorization_sink_dim: 16`, `_num_molecules: 35025`, `_start_epoch: 4` | Does removing the incentive (I1) help, whatever the route? |

**The memorization sink (`m1`)** gives each molecule its own random-effect vector `E[m]`: 16
dimensions, zero-initialized, with the model's weight decay (0.01) as its prior. It enters each
candidate's residual as `Δlogit_c = E[m]ᵀ W h_c`, where `h_c` is the contextualized token and `W`
is a learned 256 → 16 map. It is applied in training only, from epoch 4 (0-based). Val, test and
inference never pass sink indices, so every val number is the shared model. And because the term
is bilinear, `E = 0` is exactly the shared model.

- **Mechanism.** The molecule-specific part of the allocation is fitted by a free per-molecule
  parameter instead of by keys in the shared weights.
- **Batch plumbing.** The batch already carries the raw `mol_id` (template `unique_id: True`).
  The table is indexed by `mol_id` directly: 35,025 rows cover every NIST20 split id, whose
  maximum is 35,024. Rows of val and test molecules are never used, so they stay zero.
- **Reading.** The difference between train cosine with the sink on and off is what it absorbed.
  If val falls, the sink absorbed transferable signal too; then delay it or raise the weight
  decay.
- **Precedents.** Random effects in mixed models; MemSinks (Ghosal, Maini and Raghunathan, ICML
  2025).

The prepared `h1`, `h2` and `u2` are orthogonal. Submit them in the same batch if they are not
already running.

Decision table:

| outcome | reading | next |
| --- | --- | --- |
| `r0` gap within 0.02 of the control, and another route turns asymmetric | rerouting confirmed; per-input fixes are futile | Step 2 on every channel, plus Step 3 |
| `r0` gap falls by 0.05 or more, with val at least control − 0.003 | the boundary block was a removable channel | `r0` becomes the base; re-map for residual routes |
| `r1` passes | a cut term that cannot name the molecule carries the transferable part | `r1` becomes the base for Step 2 |
| `r2` gap falls on every route, with val at least control − 0.003 and `wrecall` held | global attention is the identity source | Step 2c |
| `k1` or `m1` passes | a route-agnostic cap works | stack with the best of `r0`/`r1` |

## Step 2: remove identity from the remaining channels

Each item is a matched arm on the best Step 1 base. Stack only items that pass. Watch K5 after
each one: once the token cannot name the molecule, the candidate set is the remaining side door.

- **2a. Tied cut term.** Replace the `CutBondEncoder` input with the bond-environment bucket
  Step 0.4 selects: one learned embedding per bucket, shared across molecules, backing off to a
  smaller radius where support is thin.
- **2b. Pools from a local encoder.** Pool the fragment and complement sides from a k ≤ 2 local
  encoder, not from the global one. Alternatively use coarse side descriptors, such as heteroatom
  counts by type and hybridization.
- **2c. A narrow global channel.** Project a molecule read-out to 8-16 dimensions, apply dropout
  and noise, and append it to every token. Nothing else in the token may be global.
- **2d. Coarse mass.** Bin `c_mass` at about 1 Da, or drop the precursor mass fraction.
- **2e. Tied H shifts** (only if Step 0.2 finds memorization in the ion/H head). The ion/H logits
  read only the local cut term, the descriptors and CE, not the full contextualized token.

## Step 3: caps that do not depend on the route

- **`v1` token bottleneck.** Put a variational layer after the token projection: `z = μ + σε` in
  training, `μ` in evaluation, KL to N(0, I) at β = 1e-3 and 1e-2 per candidate. It caps what
  any combination of blocks can pass per candidate. The set attention can average the noise back
  into a molecule identity, but not into a candidate identity, and the lookup needs both.
- **`s1` set dropout.** Hide 20-30% of each molecule's candidates from the attention keys in
  training, with the probe's `set_drop` code. Run it only if K5 grows after Step 2. (Built
  2026-09-27 as `sd1`, next to the message mask `sm1`;
  [section](../../history/masster_memorization_route_plan_log.md#set-dropout-on-the-allocators-keys-arm-sd1-built-2026-09-27-not-submitted).)
- **`c1` coherent gradients.** Combine the gradients of 4 molecule-disjoint micro-batches by
  elementwise median or a sign-agreement mask, which suppresses molecule-specific directions
  wherever they live (Chatterjee, ICLR 2020). It needs manual optimization and costs 1.3-2x the
  time. Run it only if `m1` and `v1` fail.

## Order and cost

1. **Local.** Build the `r0`, `r1`, `k1` and `m1` keys, with tests and smokes (**done
   2026-09-24**). Run Step 0: 4-6 probe runs at about 45 min each, plus the census (to do; it can
   run while batch 1 trains).
2. **Fir batch 1.** Submit `c0`, `r0`, `r1`, `r2`, `k1` and `m1` on `fir:h100-3g.40gb`, 12 h
   boxes, plus `h1`, `h2` and `u2` if they are pending. Ready: generate with
   `submit_masster_mh_ft_memorization_route.sh all` on Fir.
3. **Offline reading** per checkpoint: val and 25.6k-train exports, `ranking_decomp.py`, the
   Step 0 route map, and the novelty bins. Then apply the decision table.
4. **Batch 2.** Run the Step 2 and Step 3 arms the table calls for. Replicate winners, then port
   them to the merged 128-epoch screen (±0.005 gate) with checkpoint upload on.

## Not in this plan

- **More data**, including NIST23 and distillation onto unlabeled structures: out of scope by
  request.
- **Blocking the boundary block as the fix.** `r0` is a diagnostic. The wiring allows the key to
  re-form from the other blocks.
- **Formula or composition columns in the token.** They make candidates more identifiable, and
  the whole formula prior is worth +0.005
  (formula baseline (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_formula_only_baseline_2026_09_23.md`)).
- **A train-set ranking loss.** The train ranking gap is 0.011.
- **Global dropout** (`drop03`, −0.0149) and **pool-only atom dropout** (`e2`/`e3`, null).
- **Reconnecting the proposal gradient**, which costs about 0.01 through a convergence delay
  (head-input asymmetry §8.2 (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_head_input_asymmetry_2026_09_22.md`)).
- **Target cleaning**, a closed null (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_target_cleaning_cross_eval_2026_09_23.md`).
- **An identity-blind lability prior in the final score**, which is the displacement plan's
  arm 2A.

## Open questions

- Literature on addressing memorization, mapped onto these arms and ranked:
  [memorization literature note](../../notes/memorization_literature_2026_09_24.md).
- Every number is from unmerged `[M+H]+`/FT, from two checkpoints with one seed. The merged
  lineage's route map is unmeasured and needs uploaded checkpoints.
- `r2` changes the proposal's inputs too, because both branches read one encoder, and
  `separate_proposal_encoder` builds its GNN with the same `mol_gnn_type`. If `wrecall` moves,
  isolating the intensity path needs a separate encoder-type key.
- Tying costs resolution. How much candidate intensity sits in well-supported environment
  buckets is what Step 0.4 measures; if it is little, 2a stops at the local encoder of `r1`.

## Still-open proposals

### Transfer-oriented training for covered ranking failures (2026-10-01, deferred)

**Status:** DEFERRED after the user's mechanism/priority review. Research/design review only;
no trainer, config, diagnostic measurement or training run. The original bounded design is
retained below for reference. The [computed-energetics gate](#computed-product-ion-energetics-information-gate-2026-10-01-proposed)
is now the first proposed check; full attention on `i3` remains an incremental option.

**Review resolution.** The train/validation gap establishes a generalization failure, but does
not establish that encouraging agreement between training chemical groups will repair it.
For the proposed head subset H, the small-step expansion is
`J = L_A + beta * L_B - beta * alpha * <grad_H L_A, grad_H L_B> + O(alpha^2)`.
Thus its leading extra effect is gradient alignment, the same broad hypothesis as the plan's
low-priority `c1` and [literature note §4](../../notes/memorization_literature_2026_09_24.md#4-suppress-per-example-gradient-directions).
MLDG, coordinate medians and sign masks are different algorithms; the overlap in premise
does not make them equivalent. The `alpha = 0` control correctly isolates the extra effect.

The [analogue analysis](../../findings/masster/masster_i1_worst_predictions_2026_09_30.md#result-the-floor-departs-from-its-analogues)
weakens this priority: 19.8% of the floor has a top-20 analogue showing the dominant pattern,
whereas 44.9% has no similar train molecule showing it. Predictions already resemble analogues
at 0.758 while targets reach 0.247. Keeping B's own target does not remove the risk that the
alignment term discounts legitimate rare directions. B also belongs to the global training
population; repeated episodes do not make it permanently unseen. Existing train ordering is
already almost fitted. These are reasons to defer, not evidence that gradient conflict causes
the failure or that every domain-generalization objective must fail.

The independently selected `fa1 - i1@63` floor contrast is +0.014 [-0.022, +0.053], and checkpoint
floors overlap at Jaccard 0.36-0.46
([comparison](../../findings/masster/masster_i1_worst_predictions_2026_09_30.md#result-i3-on-the-floor-follow-up-2026-09-30)).
The subtraction direction was corrected in the 2026-10-02 precision audit; it still spans zero.
A costly pilot on this small, unstable endpoint could remain inconclusive. `i3`, full attention
and checkpoint averaging have no demonstrated independent-floor recovery. The additive
cut probe null (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_cut_intensity_probe_2026_10_01.md`) is another
negative result for an available correction, but does not rule out every transferable feature.

**Technical qualification.** Current `SpectrumPL` rejects manual optimization and the MASSter
batch sampler requires automatic optimization. A differentiable virtual-update loss could
in principle return a scalar under automatic optimization, so manual optimization is not a
mathematical requirement of MLDG. No such path is implemented. Differentiating through the
inner gradient still requires higher-order derivatives and retained activations; compatibility,
memory and runtime remain unmeasured. Do not assert a fixed 2-3x cost without pricing it.

**Evidence and hypothesis.** The [worst-spectrum finding](../../findings/masster/masster_i1_worst_predictions_2026_09_30.md)
puts 88% of floor spectra in the candidates-present population. Train ranking failures are
almost absent, whereas validation ranking failures are common. Cut-score addition, generic
post-cut descriptors, the small retained-fragment branch, and candidate-graph restriction did
not establish a remedy. Another within-spectrum ranking term has weak justification when the
training ordering is already fitted. The original proposal's hypothesis was that the optimizer should favor
updates whose benefit transfers across chemical groups. Cross-group gradient conflict has
not been measured; it is not an established cause of the observed errors.

**Literature scope.** [MLDG](https://arxiv.org/pdf/1710.03463) evaluates a virtual update on one
training domain using loss from another, then optimizes the combined objective. Its experiments
are image classification and reinforcement learning, not mass spectra. [Fish](https://arxiv.org/abs/2104.09937)
offers a first-order approximation to cross-domain gradient matching. Neither establishes a
MASSter gain. [DomainBed](https://arxiv.org/abs/2007.01434) found carefully implemented ordinary
training highly competitive, so a matched ordinary-training control is essential. This develops
the plan's unbuilt gradient-agreement idea, not the reverted `cm1` relationship-target recipe.

**Retained proposed adaptation, deferred.**

1. Construct chemical groups from training structures only, using scaffolds or fingerprint
   clusters. Random batches are not automatically domains. Check group sizes, molecular novelty
   and acquisition support before choosing the grouping. Within an episode, A and B contain
   different identities and chemical groups, with comparable instrument/NCE composition.
   Canonical validation/test labels never enter optimizer losses or group construction.
2. Keep `i3`'s model, primary spectral loss, support construction and EMA policy. Take a virtual
   update on the intensity allocator and boundary reader using A. Evaluate B's measured
   spectrum loss under that update. B keeps its own measured labels, including genuine
   fragmentation differences; no neighbour's spectrum or latent fragment identity becomes its target.
3. Optimize `J = L_A(theta) + beta * L_B(theta'_H, theta_rest)`, where
   `theta'_H = theta_H - alpha * grad_H L_A` is a temporary head update. Differentiate through
   that update for the MLDG-style pilot. A detached inner gradient changes the method and must
   not be presented as the same objective. Inference uses the shared unadapted weights.
4. Build a feasibility diagnostic before a campaign: price memory/time and characterize
   cross-group update transfer on training-only samples. A trained-checkpoint assay does not
   make those identities unseen and cannot establish generalization. Use a fresh bounded
   training pilot for efficacy. Existing late memorized heads are not the only intended start.

**Controls and implementation constraints.** Compare against the same A/B sampler, exposures,
loss weighting and outer optimizer with `alpha = 0`: `L_A + beta * L_B`. A random identity-group
control, matched in acquisition distribution, tests whether chemical grouping matters. Preserve
all candidate outputs and ambiguity-preserving summed-mass spectral supervision; never select
one target-compatible structure as ground truth. Preserve the support and stochastic realization
when comparing original versus virtual head weights. Run precursor encoding/proposal once per
batch, avoid extra BatchNorm updates and restore all virtual state before the actual optimizer,
EMA and checkpoint operations. The current `SpectrumPL.training_step` supports automatic
optimization only; this is not a drop-in optimizer switch. Feasibility and compatibility need
checking before a production implementation. Report both matched exposure and wall-clock cost.

**Acceptance and stop rule.** The efficacy question is held-out molecule ranking, not train fit
or gradient alignment. Use instrument-tolerance cosine, covered base-peak hits/top-three mass
ranking, unfamiliar molecules and a floor selected by a separate checkpoint. Keep the primary
spectral objective and all-molecule evaluation. A whole-validation gain with no independently
selected floor benefit does not meet this task's objective. Confirm a promising floor and
ranking effect with paired seeds and molecule-cluster intervals; predeclare numerical gates for
the actual pilot population before fitting. Do not convert the existing binned +0.005 tail gate
into an instrument-tolerance or small-floor effect threshold. Stop this design if it only reduces
train fit, improves a gradient statistic, or erases genuine rare fragmentation choices.

**Cheap transfer-premise check, deferred.** If this route is reconsidered, first characterize
head-gradient norms, cosine/sign conflict and sensitivity to grouping on training-only batches,
matched for acquisition. A fitted-checkpoint assay is a diagnostic, not unseen-molecule efficacy.
Only a substantive signal would justify a first-order experiment before a meta-gradient pilot.
[RM3](https://arxiv.org/abs/2003.07422) reuses the current and previous two gradients without an
extra forward/backward pass, but adds gradient storage and temporal mixing; it is not an exact
first-order implementation of MLDG. No diagnostic or optimizer change is requested now.

### Computed product-ion energetics information gate (2026-10-01, proposed)

**Status:** ACTIVE, feasibility and multiple-cut/full-shortlist continuation completed
2026-10-02 after the user's instructions to proceed. [Results and limits](../../findings/masster/masster_energetics_feasibility_2026_10_02.md):
joint search plus original states covers 268 / 276 covered floor base peaks; 245 survive
top-128 selection. Complete frozen shortlists/channels exist. The harder geometry sample has
297 / 322 converged jobs but only 14 / 24 distance-preserved product pairs. No spectral fitting
or ranking gate yet. No model/training code, config, original checkpoint/dataset or conda
environment changed by this audit. The original gate design follows.

**Objective and premise.** Test whether computed candidate-specific physics supplies incremental
held-molecule ranking information beyond frozen MASSter scores and existing descriptors. It is
a plausible way to address wrong-side and wrong-parity choices without relying on analogue
consensus. A benefit for the 44.9% no-similar-analogue class is a hypothesis, not a measured result.
The [residual finding's limits](../../findings/masster/masster_i1_residual_structure_2026_09_28.md#limits)
leave computed proton affinity/BDE untested. The [post-cut gate](../../findings/masster/masster_i1_postcut_information_gate_2026_09_29.md)
already found little incremental value from generic charge/structural descriptors, so the bar
is additional information, not another descriptor inventory.

**Scope and counterevidence.** Start by pricing xTB on valid candidate ions and complementary
neutral products; this is not full fragmentation dynamics or a new architecture. The
[existing literature review](../../notes/intensity_displacement_literature_2026_09_24.md#5-things-the-literature-argues-against)
already cautions against proton affinity alone and neutral homolytic-BDE predictors for
`[M+H]+`. The [CIDMD protonation study](https://pubmed.ncbi.nlm.nih.gov/39329341/) found proton
affinity a poor predictor of spectral agreement and emphasized protomer-dependent pathways.
Retain multiple plausible protomers and product states. Static stability and reaction energy
are proxies; they do not supply activation barriers or a kinetic intensity law.

**Bounded gate design, to finalize before any execution.**

1. **Feasibility and shortlist.** Use the covered floor plus controls matched for instrument,
   NCE, molecular size, candidate count and novelty, with training molecules to fit the probe.
   Fix a target-independent top-K channel shortlist from frozen scores/structural criteria;
   choose K on training-only feasibility evidence. Never insert the measured winner into the
   shortlist. Report target-mass retention and construction failures against the full eligible
   population: a null on a shortlist excluding the winner cannot diagnose energetics.
2. **Valid chemistry.** Construct ion and neutral states with balanced atoms/H, charge and
   electron parity; retain alternative compatible structures rather than assign a mass one
   unique ion. A severed precursor graph is not automatically a chemically valid product.
   Audit optimization-induced rearrangement/dissociation and formula/mass preservation.
   Record unresolved channels, especially wrong-parity cases, rather than silently excluding them.
3. **Comparable energies and cost.** Specify geometry/conformer/protomer enumeration, method,
   convergence, gas-phase conditions and cache identities. xTB accepts explicit charge and
   unpaired-electron settings ([official documentation](https://xtb-docs.readthedocs.io/en/latest/sp.html#charge-and-multiplicity)).
   Compare balanced channel energies such as
   `Delta E = E(product ion) + sum E(neutral products) - E(precursor ion)`.
   Raw total energies of different elemental compositions cannot rank competing peaks.
   Label electronic-energy differences as proxies; claiming enthalpic PA/BDE requires the
   corresponding definitions/corrections. Price time, failed calculations and state multiplicity
   on a small audit before committing to the floor set; a days-long null is not yet established.
4. **Incremental test and controls.** Preserve the frozen candidate pool and summed-mass
   supervision. Fit a simple residual/ranking probe on training molecules, calibrate on separate
   molecules, then lock the transformation and strength before confirmation. Compare frozen
   scores, scores plus existing descriptors/context, the same probe plus energies, and
   within-spectrum shuffled energies. Use fixed molecule-separated tuning/confirmation groups;
   this validation set was previously researched and is not pristine test data. Canonical test
   labels remain untouched. Computed features must not depend on measured peak identities.
5. **Promotion and stopping.** Use instrument-tolerance cosine, covered base-peak/top-three
   mass ranking, the no-analogue and wrong-side/parity strata, matched controls, and a floor
   chosen by an independent checkpoint. Retain all-molecule/no-harm accounting and molecule
   bootstrap intervals. Establish the smallest useful effect, supported sample size and numeric
   gate before fitting; the present small floor may only resolve large effects. Promote only
   incremental confirmation evidence beyond descriptors and shuffled controls. Separate a
   statistical null from inadequate precision or chemistry/shortlist failure. No architecture
   branch follows from a qualitative energy correlation alone.

**Coverage work as a separate parallel priority.** The
[O-transfer gate](../../findings/masster/masster_i1_worst_predictions_2026_09_30.md#why-the-nitro-and-sulfonyl-groups-fail-follow-up-2026-09-30)
passed for `substituted_aryl_sulfonamide_o_transfer`; the existing
[`so1` templates-on config](../../../config/masster/mh_ft_memorization_route_2026_09_24/so1_smarts_o_transfer_ob_obf.yml)
needs its run state checked before proposing another submission. Read eligible molecules and
the 151 sulfonyl tail spectra against fixed and independent selectors; coverage evidence alone
does not establish correct allocation to the new masses. The arm enables all registered
templates, so attribution to this one rule needs candidate-source accounting. Separately, a
[4x candidate-budget replay](../../findings/masster/masster_i1_worst_predictions_2026_09_30.md#result-uncovered-candidates)
improved coverage in all nine sampled selection misses (0.23 to 0.71). This is a positive local
gate, not a demonstrated full-split intensity gain or justification to enlarge every pool.
No coverage run is launched as part of this review.

**Current state and next concrete check (2026-10-02):** the joint-state search and complete
frozen shortlists are available; formal construction no longer excludes most covered winners.
Set a bounded, target-independent alternative-state policy and calculation budget before
constructing energy features on training/calibration and confirmation molecules. The full
audit shortlist contains 199,903 unique molecular channels, so the small pricing sample does
not justify treating full enumeration as cheap. Keep unresolved geometries/features in the
population, with zero correction or explicitly calibrated missing-feature handling; do not
drop their spectra. Retain all stored latent alternatives even if only a fixed predicted-share
subset receives calculations. Use the fresh complete-channel production baseline: the older
exporter's many-to-one decomposition can inflate a support oracle above one.

Predeclare the independent selector, molecule partitions, useful effect and precision gate
before fitting a correction. Test energies/status flags incrementally beyond existing
descriptors and shuffled controls. Geometry changes are unvalidated flags, not automatically
useful physics or evidence of invalid fragmentation. No information-test null or gain exists.
Ensembling remains last. The isolated checksum-verified xTB executable and source snapshots
are in ignored audit artifacts; the conda environment is unchanged. See the finding for
coverage, cost, geometry flags, replay differences and independent validation.
