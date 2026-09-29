# Closing MASSter's per-molecule allocation lookup on every route

> **Status:** ACTIVE · **Updated:** 2026-09-29
> **2026-09-29: [Batch 5 read](#batch-5-read-from-wb-i3-i4-fa1-and-mz1-2026-09-29)** (`i3`, `i4`,
> `fa1`, `mz1` finished). `i3` (`k1` + EMA on `i1`) passes at +0.0102, and `i4`'s pre-norm adds
> +0.0019. Full attention (`fa1`) is +0.0044 over `i1`, just under the one-seed rule, with val loss
> better and train fit lower. The m/z encoding (`mz1`) is rejected: −0.0017 against `fa1`, ahead
> only early. Arms: [`fa1`/`mz1`](#full-attention-and-the-mz-encoding-on-i1-fa1-and-mz1-built-2026-09-29-not-submitted),
> [`i3`/`i4`](#stacking-on-the-instrument-input-i3-and-i4-built-2026-09-28-not-submitted).
> **2026-09-28: [Batch 4 read](#batch-4-read-from-wb-2026-09-28)** (`s1`, `s1_k1_ema`, `m2`, `sm1`,
> `sd1`, `am1` finished): the cosine schedule passes against `c0` (+0.0090) but adds nothing on
> top of EMA; the rest are single-seed or null. **Stacking arms `i3`/`i4` built on `i1`**, the
> instrument-input arm ([section](#stacking-on-the-instrument-input-i3-and-i4-built-2026-09-28-not-submitted)).
> **2026-09-28: [memorization read on `i1`](#memorization-read-on-i1-2026-09-28).** The
> instrument input narrows the eval-mode gap from 0.183 to 0.158 at epoch 63 while train fit
> rises. About 86% of the gap remains.
> **Audited 2026-09-27 ([finding](../../findings/masster/masster_memorization_attempts_audit_2026_09_27.md)).**
> Every number below was re-derived from W&B and code. The corrections are applied in place. The
> +0.003 gate is about one standard error of a single-seed delta, and gates 2-4 have not run on
> any arm.
> **The allocator-route probe reran on `c0`, `r2`, `nl1` and `nl2` (2026-09-27).**
>
> - On `c0` the allocator's set message is the most train-specific route (4.2x), as on f0.
> - Without global attention (`r2`) it carries almost twice the train-specific fit: this is where
>   the lookup goes.
> - Two masks on that route are running against `c0` since 2026-09-27 (`7kz04hra`, `uxie2eko`):
>   - the message mask `sm1`
>     ([section](#set-message-substitution-arm-sm1-built-2026-09-27-not-submitted));
>   - the key mask `sd1`
>     ([section](#set-dropout-on-the-allocators-keys-arm-sd1-built-2026-09-27-not-submitted)).
> - Proposed, not built: the candidate-graph arms `fa0`/`cg1`/`cg2`, the user's GNN over the
>   candidates, as full attention masked to chemical neighbours
>   ([section](#candidate-graph-allocator-arms-fa0-cg1-and-cg2-proposed-2026-09-27-not-built)).
> - Built, not run: the restricted allocators. They are a mass window on full attention, the MLP
>   (`attention: none`) and smaller induced sets, with a seven-run local queue
>   ([section](#restricted-allocators-mass-window-no-attention-smaller-set-transformer-built-2026-09-27-not-run)).
>
> **Step 1 ran; W&B tail read done ([below](#step-1-tail-read-from-wb-2026-09-24)).** `k1` (key
> dropout) passes gates 1 and 5 at +0.0074 over `c0`; gates 2-4 need the route map and novelty
> bins on its checkpoint. `r2` (no global attention) is null at +0.0006. `r1` misses the gate
> (−0.0029, inside the seed spread), `r0` is −0.0129 (with a changed initialization) and `m1` is
> −0.0203. The same-recipe `c0`/`f0` spread is 0.0022.
> **Global-message noise arms `gm1`/`gm2`**
> ([section](#global-message-noise-arms-gm1-and-gm2-built-2026-09-25-not-submitted)) **and swap
> arms `gm3`-`gm6`**: another molecule's real message instead of the batch mean, a feature-level
> CutMix that keeps the label
> ([section](#global-message-swap-arms-gm3-gm6-built-2026-09-25-not-submitted)).
> **Batch 2** ([section](#batch-2-configs-2026-09-25-not-submitted)): `gm1`-`gm6`, `k1_ema` and
> seed-811237 replicates of `c0`, `k1` and EMA. **Six of ten finished; tail read 2026-09-26**
> ([read](#batch-2-tail-read-six-finished-runs-2026-09-26)). The seed spread is 0.0042
> (`c0_s2` − `c0`), above the +0.003 gate. `k1` and EMA replicate (+0.0048, +0.0072 at seed
> 811237); `k1_ema` is +0.0029 over `u2`, just under the gate; `gm1` +0.0051 (one seed), `gm2`
> −0.0029. No swap arm `gm3`-`gm6` passes (+0.0023 to −0.0029, each inside the seed spread), and
> each is below `gm1`.
> **Learning-rate-schedule arms `s1`/`s1_k1_ema` built 2026-09-25, not submitted**
> ([section](#learning-rate-schedule-arms-s1-and-s1_k1_ema-2026-09-25)): `d3`'s warmup and cosine
> decay on `c0` and on `k1_ema`, where every run so far trains at a constant 1e-3 (decision C).
> **Batch 3 read 2026-09-26 ([read](#batch-3-read-from-wb-2026-09-26)).** Weight-decay arm `w1`
> (`weight_decay` 0.1) finished null at +0.0010. **Light-side token arms, final read 2026-09-27**
> ([read](#final-read-of-nl1-and-nl2-2026-09-27)):
>
> - `nl1` finished at +0.0037, an unresolved single-seed pass, not the projected null.
> - `nl2` finished at −0.0015, and −0.0021 against its same-code control `r2` with train and val
>   loss worse: a mild underfit.
> - `nl3` underfits (−0.018 against `nl2`) and crashed at epoch 30.
> - Neither `nl1` nor `nl2` touches the boundary block, `c_mass` or the candidate set, so neither
>   can show rerouting.
>
> `sp0`/`sp1` collapsed on the broken sampler and were resumed from their epoch-2 checkpoints
> after the fix
> ([shared-proposal audit](../../findings/masster/masster_shared_proposal_pathway_audit_2026_09_26.md)).
> Sink-dropout arm `m2` is running at +0.0035 over `c0` (epochs 31-40).
> ([`w1` section](#weight-decay-arm-w1-2026-09-25),
> [`nl` section](#light-side-token-arms-nl1-nl3-built-2026-09-26-not-submitted).)
> **Step 0's route map ran (items 1-3):** the global-attention message in the encoder writes the
> lookup key, and the boundary block reads it
> ([finding](../../findings/masster/masster_route_map_2026_09_24.md)). The collision census
> (item 4) is not built. The consolidated [change list](#change-list-2026-09-24) records every
> planned change and the three open decisions (EMA in every arm; replacing `k1` with a
> global-message regularizer; a learning-rate-schedule arm). MASSter only; every fact below is
> from MASSter code or runs.
> The [memorization probe](../../findings/masster/masster_memorization_route_probe_2026_09_23.md)
> found the lookup in the boundary block. But every candidate input MASSter builds comes from
> the same globally attended atom embeddings, so blocking that one input can move the lookup
> instead of removing it. This plan first maps every route by which the lookup can reach the
> output, including two the probe cannot see: the encoder, and the ion/H head. It then removes
> molecule identity at its source, or caps what passes after the token, and re-maps the routes
> after each change. It complements the [ranking-transfer](masster_ranking_transfer_plan.md)
> arms `h1`, `h2` and `u2`. The identity-blind lability prior in the final score belongs to the
> [intensity-displacement plan](masster_intensity_displacement_plan.md) (its arm 2A). Adding
> data is out of scope.

## Objective and acceptance

**Review, 2026-09-27:** the [plan review](../../findings/masster/masster_memorization_plan_review_2026_09_27.md)
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
  ([ranking decomposition](../../findings/masster/masster_ranking_decomposition_2026_09_23.md)).

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
   ([variance-novelty](../../findings/masster/masster_variance_novelty_2026_09_24.md)).
5. **Coverage.** `wrecall` stays within 0.005 of the control.

Narrowing the gap by lowering train fit is underfitting, not a pass. `drop03` (−0.0149) did
that, with train cosine logged. `b3_wd_groups` and `e2` logged no train cosine, and narrowed only
the train-mode loss gap; `b3`'s change is inside that statistic's own floor.

## Change list (2026-09-24)

Everything this plan will change in MASSter, by status. Evidence is from the
[route map](../../findings/masster/masster_route_map_2026_09_24.md), the
[ranking-transfer tail read](masster_ranking_transfer_plan.md#tail-read-from-wb-2026-09-24),
and the [memorization literature note](../../notes/memorization_literature_2026_09_24.md).

**Built and tested, not submitted.** Each arm is one change against the control `c0`.

| arm | change | why |
| --- | --- | --- |
| `r2` | encoder without global attention (`mol_gnn_type: GINE`) | Global attention writes the key: train-to-val damage 2.6x against 1.2x for the generalizing routes. Graph-transformer "over-globalizing" (ICML 2024) is the same effect. Strongest evidence. |
| `r1` | boundary block built from a one-hop local encoder with no global attention | The cut term keeps its chemistry but cannot name the molecule. |
| `r0` | boundary block removed | Diagnostic only: does the lookup move to another input? |
| `m1` | memorization sink: a 16-dim per-molecule vector, training only, dropped at evaluation | Removes the incentive to build keys. Precedents: MemSinks (ICML 2025), example-tied dropout, random effects for repeated measures (LMMNN). |
| `k1` | dropout 0.2 on the cut-bond encoder and the projected token | **Weakened**: the key is written upstream in the encoder, and `h1` (allocator dropout) was rejected at −0.0067. |
| `c0` | control: `f0` plus candidate-coverage logging | Same-code reference. |

**Decisions before submission (open):**

- **A. Add `ema_decay: 0.999` to `c0` and every arm.** EMA passed on both lineages (+0.0094
  unmerged, +0.0097 merged). The weight-averaging literature (SWA, SWAD) supports it.
- **B. Replace `k1` with a global-message regularizer.** Four options, all new code. The first
  two are built as `gm1`/`gm2` (2026-09-25), with dropout drawn per atom rather than per molecule;
  see [the section](#global-message-noise-arms-gm1-and-gm2-built-2026-09-25-not-submitted).
  A fifth, swapping in another molecule's real message, is built as `gm3`-`gm6`, including the
  per-molecule draw ([section](#global-message-swap-arms-gm3-gm6-built-2026-09-25-not-submitted)).
  - **global-message dropout**: in training, replace the global-attention message with the batch
    mean for a random share of molecules;
  - **global-message noise**: Gaussian noise on `gsa_out` in training. Noise injection is
    Tikhonov regularization and penalizes high-frequency functions, which is what a lookup is
    (Bishop 1995; Camuto 2020);
  - **gated global message**: a learned sigmoid gate on `gsa_out`, so an atom can attend to
    nothing (SigGate-GT, 2026), optionally with a sparsity penalty;
  - **DropKey on the GSA layers**: drop keys before the softmax (DropKey, CVPR 2023), so each
    atom's global message comes from a random subset of the molecule's atoms. It only weakens
    identity, where global-message dropout removes it.
- **C. Add a learning-rate-schedule arm `s1`: `c0` plus the `d3` schedule** (cosine decay,
  `lr_min_ratio: 0.01`, 1000-step warmup, `lr_total_steps` = 64 epochs × steps per epoch). Every
  MASSter run so far, the Step 1 arms included, uses a constant `lr: 1e-3`. `d3` gained
  +0.0072 on the merged lineage (n=1). A large initial rate followed by decay keeps a network
  from memorizing an easy-to-fit patch first (Li, Wei & Ma 2019; You et al. 2019), and a
  molecule key is such a patch. Recommendation: run it as its own arm and read the gap
  trajectory, rather than putting the schedule in every arm. The route map was measured on
  constant-rate checkpoints, and the other arms should stay matched to them. **Built
  2026-09-25** as `s1`, with a stacked `s1_k1_ema` against `k1_ema`
  ([section](#learning-rate-schedule-arms-s1-and-s1_k1_ema-2026-09-25)).

**Next, ranked by evidence (not built):**

1. Sink refinements for `m1`: turn the sink off on part of each batch, as MemSinks mixes shared
   and specific units, and learn its shrinkage like a random-effect variance instead of fixing
   weight decay. The concrete form is a variational sink: a Gaussian posterior per molecule with
   a KL term to N(0, σ²I) and σ learned (GLMMNet, MC-GMENN and MMbeddings all beat plain entity
   embeddings for high-cardinality IDs).
2. **Goldfish-style masked peaks** (Hans et al., 2024). Hash (molecule, peak) to exclude about 25%
   of each molecule's target peaks from every exposure. Drop them from both the target and the
   predicted distribution, so the loss does not teach that they are absent. No lookup can then
   reproduce a full allocation.
3. **Label smoothing toward a smooth prior** (uniform over candidates, or the formula prior). It
   targets the measured over-sharpness (entropy 0.12-0.15 nats low; `P^0.75` +0.007) as well as
   memorization. Data-dependent priors beat uniform ones, and an optimal strength exists, so run
   two strengths. Keep it out of any distillation teacher: a smoothed teacher distills worse.
4. RM3 (coordinate-wise median of the current and two previous gradients; no extra passes),
   AND-mask, or gradient agreement filtering (average only the micro-batches whose gradients
   agree by cosine; Chaubard et al., 2024) over molecule-disjoint micro-batches, as the
   route-agnostic optimizer arm.
5. A variational bottleneck on the global message, which is more targeted than the token-level
   `v1`. The nonparametric VIB is built for attention: it limits how many vectors attention can
   reach and how much each carries.
6. Later options:
   - SAM for only the last 16-32 epochs, from a checkpoint (fine-tuning with SAM alone gives
     significant gains; full SAM is about 2x, ESAM about 1.4x);
   - a shard soup: branch K runs from an early checkpoint (about epoch 8), train each on a
     molecule-disjoint shard, and average the weights, so each molecule's lookup is diluted to
     1/K (Diffusion Soup, 2024; bounded by our measured +0.003 weight average and +0.020
     ensemble);
   - out-of-fold distillation (never same-data: soft labels carry a teacher's memorized
     answers);
   - targeted larger weight decay over a longer horizon (the grokking and Omnigrok line;
     `w1` tests 10x decay at 64 epochs, [section](#weight-decay-arm-w1-2026-09-25));
   - a Fisher-trace penalty (a second backward pass);
   - periodic reinitialization of the key reader or of the global-attention projections.
     Demoted: across 15,000 models, resets added little once other regularization was tuned
     (Zaidi et al., 2022). The cheap form resets the online weights to the EMA weights.

**Diagnostics to add (no training):**

- Layer rewinding: revert one module to an early-epoch checkpoint and read the gap. Literature
  disagrees on whether memorization sits in late layers or in a few neurons spread across
  layers. Needs a run that saves checkpoints at epochs 4, 8 and 16.
- An activity-cliff pair metric: near-identical molecule pairs with dissimilar spectra. It
  detects copying of the nearest training neighbour; val molecules above 0.9 Tanimoto score
  below the 0.8-0.9 bin. Read it per near-duplicate cluster: near-duplicates hide memorization
  from per-sample influence scores.
- A per-molecule memorization score: loss curvature averaged over epochs (one run suffices), or
  the norm of each molecule's `m1` sink vector, which is free once `m1` has run.
- Gradient coherence (m-coherence) per epoch on a fixed molecule sample, to see when updates stop
  being shared.
- For every arm, read the Tanimoto bins at or below 0.6 as well as overall val. In-distribution
  performance predicts similarity-cluster performance poorly (r ≈ 0.4), so an arm that misses
  criterion 1 but wins those bins clearly gets a replicate, not a rejection.

**Conditional on Step 1 results:**

- If `r1` or `r2` wins: pools from a local encoder, a narrow noisy global channel (8-16 dims),
  and cut-environment buckets shared across molecules (after the collision census).
- If `r0` shows the lookup rerouting: the token bottleneck `v1` and training-time set dropout.

**Not changing:** tying the ion/H head, coarsening mass, discretizing `a_charge` (all small or
null in the route map); generic dropout; DropEdge (reopened 2026-09-27 as `de1`,
[section](#encoder-atom-mask-and-bond-dropout-arms-am1-and-de1-built-2026-09-27-not-submitted));
DP-SGD-style per-molecule clipping (costly,
and it hurts rare examples); co-teaching (built for noisy labels); mixup across molecules;
excluding low-loss molecules without a memorization score (low train loss does not separate
memorized molecules from easy ones); same-data self-distillation; smaller batches (sync-bound);
blocking the boundary block alone as a fix;
formula columns in the token; a train-set ranking loss; reconnecting the proposal gradient; more
data (out of scope).

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
| K4 | **Exact descriptors and inputs.** `c_mass` (log1p neutral mass plus precursor mass fraction) nearly fixes the formula. `c_charge` and the `a_charge` atom input are continuous Gasteiger values. | Probe: descriptors used symmetrically. Removing `c_mass` +0.0012, `c_charge` −0.0012, `a_charge` −0.0025, all inside the merged ±0.005 gate ([campaign read-out §8](../../findings/masster/masster_campaign_readout_2026_09_21.md#8-the-bundles-leave-one-out-arms-read-at-the-128-epoch-screen-added-2026-09-24)). | partly |
| K5 | **The candidate set.** In evaluation, the 256 selected fragments are a fixed function of the molecule, and the allocator attends over all of them. | Probe: `set_drop` closes 9-10% of the gap. Training already perturbs the set through BatchNorm, and the lookup forms anyway. (A random-weight probe measured train-versus-eval Jaccard 0.60-0.77 at dropout 0.1; the trained magnitude is unmeasured, [regularization audit §2.2](../../findings/masster/masster_regularization_audit_2026_09_19.md).) | yes |
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
   `scripts/debug/debug_pooled_embedding_memorization.py`:
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
   [cut-descriptor finding](../../findings/masster/masster_candidate_cut_descriptors_2026_09_22.md)
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

**Result, 2026-09-24 ([route map](../../findings/masster/masster_route_map_2026_09_24.md)).**
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
  [section](#set-dropout-on-the-allocators-keys-arm-sd1-built-2026-09-27-not-submitted).)
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
  ([formula baseline](../../findings/masster/masster_formula_only_baseline_2026_09_23.md)).
- **A train-set ranking loss.** The train ranking gap is 0.011.
- **Global dropout** (`drop03`, −0.0149) and **pool-only atom dropout** (`e2`/`e3`, null).
- **Reconnecting the proposal gradient**, which costs about 0.01 through a convergence delay
  ([head-input asymmetry §8.2](../../findings/masster/masster_head_input_asymmetry_2026_09_22.md#82-the-71-deficit-is-a-convergence-delay-not-a-lower-endpoint)).
- **Target cleaning**, a [closed null](../../findings/masster/masster_target_cleaning_cross_eval_2026_09_23.md).
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

## Step 1 implementation and validation (2026-09-24)

In the working tree, uncommitted. Every option is off by default. At its default the model is
unchanged and draws no extra random numbers. The two new modules are built on a forked RNG, so
enabling one leaves every other parameter's initialization exactly as in `c0`.

### Code

- `CutBondEncoder(dropout=...)` in `base_model.py`: hidden-layer dropout, applied in `forward` so
  the state-dict keys, and old checkpoints, are unchanged. FraGNNet's `cut_bond` instance keeps 0.
- `CandidateIntensityTransformerHead(input_dropout=..., sink_dim=...)` in `bond_fragment_heads.py`:
  dropout on the projected token, a `sink_projection` map, and a keyword-only `sink_vectors`
  forward argument. The positional arguments are unchanged, so the probe's patch still applies.
- `MassterModel` in `masster_model.py`:
  - `candidate_boundary_block`;
  - `boundary_local_layers`, which builds a separate plain-GINE `boundary_local_encoder`;
  - `boundary_encoder_dropout` and `intensity_input_dropout`;
  - `memorization_sink_dim` and `memorization_sink_num_molecules`, which build a zero-initialized
    `memorization_sink` table;
  - the `memorization_sink_indices` forward argument.

  Inconsistent combinations raise.
- `MassterPL._memorization_sink_indices` in `masster_pl.py` passes the batch's `mol_id`
  only in training, from the start epoch. Evaluation and inference get none.
- `config/template.yml` declares and comments the seven keys.

### Validation

- `tests/test_masster_memorization_route.py`, 27 tests:
  - state-dict keys are unchanged, nothing draws random numbers at the defaults, and both dropouts
    act in training only;
  - the boundary block is zero in both modes, and the combinations that act on it are rejected;
  - the local encoder is plain GINE, it feeds the `CutBondEncoder`, and it leaves every other
    initialization unchanged;
  - the sink starts as the shared model with bit-identical outputs, moves only its own molecule's
    spectrum, is unused without indices, receives gradient only for the batch's molecules, and
    rejects bad indices and incomplete configs;
  - the PL passes indices only while the sink is training.
- The existing cut-bond, candidate-feature, probe, MASSter model and PL tests pass (413).
  The full suite passes: 4,327 passed, 1 skipped, 2 xfailed. Ruff is clean on every changed
  Python file.
- **Smokes.** Two epochs on `debug_tiny_masster_cut_features.yml`, one per arm option, on the
  local RTX 2070 SUPER. All exited 0 and trained and validated both epochs.
  - Debug-scale parameter counts move as intended: control and `k1` 340K, `r0` 324K, `r1` 354K,
    `r2` 307K, `m1` 902K.
  - The `m1` checkpoint holds a 35,025 × 16 table. Only the rows of the 32 train molecules seen
    in the sink epoch moved; all 40 val/test-only rows are exactly zero. It reloads with
    `strict=True` through `init_model_from_ckpt`, in eval mode.
  - The two-epoch val cosines are noise and are not read.
- **Production parameter counts**, from the resolved configs built on CPU:

  | config | total | encoder | `CutBondEncoder` | local encoder | sink | head |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | `c0` | 9.213M | 1.809M | 0.787M | — | — | 2.457M |
  | `r0` | 8.426M | 1.809M | — | — | — | 2.457M |
  | `r1` | 9.541M | 1.809M | 0.787M | 0.328M | — | 2.457M |
  | `r2` | 8.029M | 0.624M | 0.787M | — | — | 2.457M |
  | `k1` | 9.213M | 1.809M | 0.787M | — | — | 2.457M |
  | `m1` | 9.777M | 1.809M | 0.787M | — | 0.560M | 2.461M |

  Global attention is 1.185M of the encoder's 1.809M.
- **Resolved-config diffs.** Each arm is `c0` plus its own keys and naming. `c0` is `f0` plus
  naming and the coverage-monitoring key.
- **Generator.** `bash -n` passes. Job generation must run on Fir, which has `wheels/`.

### Next

1. On Fir, run `submit_masster_mh_ft_memorization_route.sh all` (`c0` plus the five arms,
   `fir:h100-3g.40gb`, 12 h each), and record the run IDs here. The coverage metrics add one
   match-mask pass per spectrum per step, equally in `c0` and the arms, so wall clock may exceed
   f0's 4-5 h; the 12 h box covers it.
2. Build Step 0 while they train, since acceptance criterion 3 reads it.
3. The `m1` sink-on train read needs the export pass to feed sink indices. That is not built.

## Step 1 tail read from W&B (2026-09-24)

All six runs finished 64 epochs in 3.2-3.3 h each, submitted 2026-09-24 21:57 UTC in W&B group
`nist20_mh_ft_memorization_route_2026_09_24`. Every run uploaded `ckpt/last.ckpt` and its best
epoch as run files, not as artifacts. Metric: `val_cos_sim_0.01_epoch/mean`, tail = mean of
epochs 54-63. "Above c0" counts the epochs at which the arm beats `c0`. The loss columns are the
54-63 mean of `coverage_weighted_spec_ce`. Train losses are logged in train mode, with dropout,
BatchNorm batch statistics and the `m1` sink all on, so the train-val loss gap is not the
eval-mode cosine gap of criterion 3.

| arm | run | tail | vs `c0` | best (epoch) | above c0 | `val_wrecall` vs c0 | train CE | val CE | gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `c0_control` | `vcpho5qm` | 0.7276 | — | 0.7292 (42) | — | 0.9396 | −3.564 | −2.881 | 0.683 |
| `r0_no_boundary` | `qyjlg9tb` | 0.7147 | **−0.0129** | 0.7158 (60) | 0/64 | +0.0004 | −3.516 | −2.826 | 0.690 |
| `r1_local_cut` | `s9p8neke` | 0.7247 | −0.0029 | 0.7258 (58) | 10/64 | −0.0005 | −3.558 | −2.864 | 0.694 |
| `r2_no_gsa` | `9lt5s2u8` | 0.7283 | +0.0006 | 0.7297 (57) | 47/64 | +0.0001 | −3.558 | −2.889 | 0.669 |
| `k1_key_dropout` | `wg0nxymi` | **0.7350** | **+0.0074** | 0.7365 (61) | 57/64 | +0.0004 | −3.484 | −2.920 | 0.564 |
| `m1_sink` | `liwjgdq6` | 0.7074 | **−0.0203** | 0.7088 (56) | 3/64 | −0.0002 | −3.591 | −2.867 | 0.724 |
| ref `f0` | `3bej6fq7` | 0.7298 | +0.0022 | 0.7318 (61) | 55/64 | +0.0005 | −3.565 | −2.889 | 0.676 |
| ref `u2_ema999` | `c8hh9j1b` | 0.7392 | +0.0116 | 0.7394 (56) | 64/64 | +0.0007 | −3.566 | −2.923 | 0.643 |

**The replicate floor is about 0.002.** `c0` differs from `f0` only in default-off keys and the
coverage-monitoring flag. The code moved between them (the `CutBondEncoder` dedup), and `c0`
sits 0.0022 below `f0`. That is the only same-recipe pair on this lineage, and it lies just under
the +0.003 gate. The `c0` epoch-to-epoch change in the tail is 0.0009.

Per arm, against the decision table:

- **`k1` passes gates 1 and 5; gates 2-4 are open.** +0.0074 is 3.4x the `c0`/`f0` spread, and
  `k1` is +0.0052 over `f0` as well. Its val CE is the lowest of the six (−2.920) and barely rises
  after its minimum (+0.005 from epoch 36, against +0.022 for `c0`). The train-mode gap falls
  0.683 → 0.564, but most of that is train loss raised by the dropout itself (+0.080), so it cannot
  rule out underfitting. The eval-mode train cosine from the route map decides that. The table
  row "stack with the best of `r0`/`r1`" has no partner, since neither passes. The stacking
  partner is the weight EMA instead (`u2`, a variance lever, whereas `k1` acts on the key).
- **`r2` is null on val with 1.18M fewer parameters.** Removing global attention (1.185M of
  the encoder's 1.809M) changes val by +0.0006, keeps `wrecall`, and moves the loss gap only
  0.683 → 0.669. So global attention is not needed for val, and removing it does not visibly
  narrow the gap. That fits the key re-forming through local message passing. The row "`r2` gap
  falls on every route" needs the route map on `9lt5s2u8`. Val CE bottoms at epoch 11, against 35
  for `c0`. `r2` has 8.029M parameters against `c0`'s 9.213M.
- **`r0` is not a removable channel.** Val is −0.0129, below `c0` at every epoch, and the
  loss gap does not narrow (0.690). Whether this is rerouting (row 1) depends on another route
  turning asymmetric, which only the route map on `qyjlg9tb` can show.
  - **Initialization differs from `c0` (audit, 2026-09-27).** `r0` also drops the
    `CutBondEncoder`, whose construction is not on a forked RNG
    ([`masster_model.py:1720`](../../../src/fragnnet/model/masster_model.py#L1720)). So
    `charge_head` and part of `intensity_head` start from different weights than in `c0`.
  - The −0.0129 is three times the seed spread, so its direction stands. Its size is not a
    clean single-variable effect.
- **`r1` misses gate 1** (−0.0029, above `c0` at 10 of 64 epochs, gap 0.694). That is inside the
  0.0042 seed spread measured later, so it shows no benefit rather than harm.
- **`m1` is rejected as configured.** −0.0203, above `c0` at 3 of 64 epochs. The sink-on train
  loss is the best of the six (−3.591) while val falls. The plan read this as the sink absorbing
  transferable signal.
  - **Not tested (audit):** a second reading predicts the same curves. Once the sink fits a
    spectrum, the shared weights get little gradient from it and train less.
  - The eval-mode train cosine with the sink on would separate the two readings. It is not
    built.
  - The plan's fallback is a later start or a larger sink weight decay. A −0.020 gap at the
    current settings is weak grounds for a retry.

### Next after the tail read

1. Run the Step 0 route map on `wg0nxymi` (`k1`), `9lt5s2u8` (`r2`) and `qyjlg9tb` (`r0`)
   against `vcpho5qm`, for gates 2-3 and the eval-mode train cosine.
2. Novelty bins (gate 4) on the `k1` checkpoint.
3. Replicate `k1` (second seed), and build a `k1` + `ema_decay: 0.999` arm on this same code.

### Eval-mode train/val gap on the Step 1 checkpoints (2026-09-25)

Item 1's eval-mode train cosine, without the route conditions. The instrument is
`debug_pooled_embedding_memorization.py`, with the settings the route map used: eval mode, 60 batches of
512 per split (about 30.7k spectra each), the same batches for every checkpoint. Each arm's best
checkpoint is used. The plain condition is read; one `feat_charge` condition ran alongside and is
not read. `c0`'s best checkpoint is from epoch 42 and the arms' are from epochs 56-61, and the
gap grows with training. So `c0` was also read at its last checkpoint (epoch 63), and f0 (`3bej6fq7`,
epoch 61, route map) is a second same-recipe reference. Runner and CSVs were in the session
scratchpad; checkpoints are in `saved_ckpts/`.

| arm | epoch | train cos | val cos | gap | gap vs `c0`@63 | train `wrecall` | val `wrecall` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c0` | 63 | 0.9222 | 0.7388 | 0.1834 | — | 0.9469 | 0.9442 |
| `c0` | 42 | 0.9162 | 0.7402 | 0.1760 | −0.0074 | 0.9462 | 0.9439 |
| f0 | 61 | 0.924 | 0.744 | 0.181 | −0.003 | — | — |
| `k1` | 61 | 0.9132 | 0.7467 | 0.1665 | **−0.0169** | 0.9469 | 0.9450 |
| `r2` | 57 | 0.9185 | 0.7420 | 0.1765 | −0.0069 | 0.9460 | 0.9445 |
| `r1` | 58 | 0.9182 | 0.7392 | 0.1790 | −0.0044 | 0.9466 | 0.9434 |
| `r0` | 60 | 0.9135 | 0.7289 | 0.1846 | +0.0012 | 0.9468 | 0.9448 |
| `u2` (EMA weights) | 56 | 0.9324 | 0.7516 | 0.1808 | −0.0026 | 0.9475 | 0.9448 |
| `m1` (sink off at eval) | 56 | 0.8142 | 0.7223 | 0.0919 | −0.0915 | 0.9473 | 0.9447 |

- **The gap is about 0.18 in every arm but `m1`.** The best arm, `k1`, closes 0.017 of it (9%):
  train −0.009, val +0.008. It narrows the gap by fitting train less and val more, which is the
  shape the plan wants, but the size is small.
- **EMA does not narrow the gap.** It lifts train (+0.010) and val (+0.013) together, so its gain is
  an optimization gain, not less memorization.
- **Removing global attention leaves the gap in place.** `r2` is −0.007, within the `c0`
  epoch-42/63 difference. The lookup the route map traced to global attention re-forms without it,
  as its Limits section allowed.
- **`m1`'s small gap is underfitting.** With the sink dropped at eval, train falls to 0.814 and
  val to 0.722: the sink carried train fit, and some transferable fit with it.
- **Coverage is not the gap.** `wrecall` differs by 0.002-0.004 between train and val in every
  arm. The gap is in how intensity is allocated among covered candidates.
- **Limits.** One seed; 60 batches, paired, so contrasts of about 0.003 are near the noise; the
  arms' best epochs differ by up to 5.
- **Audit caveats (2026-09-27).**
  - **The val sample is not random.** The probe reads the first 60 of 102 val batches through
    `SequentialSampler`, and the val split is ordered by `spec_id`. Its val cosine is
    0.010-0.014 above the full-split W&B value for every checkpoint above (for example `c0`@63:
    0.7388 against 0.7279). The train sample is 60 random batches. Absolute gaps are therefore
    about 0.011 too small.
  - **The control was read at other epochs.** `c0`'s gap grows 0.1760 → 0.1834 between epochs
    42 and 63. Interpolated to each arm's epoch, `k1` closes about 0.015 (not 0.017) and `r2`
    about 0.005 (not 0.007).
  - **Not reproducible.** The CSVs were in a session scratchpad and are gone.

The running batch-2 arms log only train-mode loss, which carries each arm's own training noise
(and, under EMA, the online weights), so their W&B loss gap is not a clean read. Val primary loss
is clean. At epochs 14-23, `gm1`'s val loss is better than `c0`'s (−2.928 against −2.889) at an
unchanged train-mode loss (−3.416 against −3.421); `gm2` raises train-mode loss (−3.356) for a small
val-loss gain. `gm1`, `gm2` and `k1_ema` upload checkpoints, so the same eval-mode read runs on
them when they finish.

## Global-message noise arms `gm1` and `gm2` (built 2026-09-25, not submitted)

**Why.** Step 1 put its one gain on noise, not removal. On the boundary path, removing the block
(`r0`) cost −0.0129 while dropout on the key-forming layers (`k1`) gained +0.0074. Removing global
attention (`r2`) held val but did not narrow the gap. The route map says the global-attention
message writes the key. These two arms put `k1`-style training noise there and keep the path.
They are the first two options of decision B.

| arm | key against `c0` | what it does in training |
| --- | --- | --- |
| `gm1_message_substitution` | `masster_global_message_substitution: 0.3` | each atom's normalized global message, in each encoder layer, is replaced by the batch mean with probability 0.3 |
| `gm2_message_noise` | `masster_global_message_noise_std: 0.5` | Gaussian noise of 0.5 × each feature's batch standard deviation is added to the normalized global message |

**Design.**

- **Where.** In `GINE_GSA.forward`, after the global message's norm and before
  `x = local_out + gsa_out` ([`nn_blocks.py`](../../../src/fragnnet/model/nn_blocks.py)). The
  norm's batch statistics therefore neither absorb the noise nor carry it into evaluation.
- **`gm1` is the route map's `gsa` popmean intervention used as training noise.** The mean is
  over every atom in the batch and detached, so a replaced atom's message gets no gradient.
- **Per atom, not per molecule.** Decision B described a per-molecule draw ("a random share of
  molecules"). The built form draws per atom, so a molecule's other atoms keep their message in
  that layer. The per-molecule form is stronger and is not built. It is the next variant if `gm1`
  is null.
- **The encoder's element-wise dropout (0.1, before the norm) is unchanged.**
- **Scope.** Only the shared precursor encoder gets the keys; a separate proposal encoder never
  does. `c0` has none, so the proposal reads the same detached embeddings and trains on perturbed
  ones. Read `val_candidate_wrecall` first.
- **Guard.** Either key set with a non-`GINE_GSA` encoder raises.
- **Neither arm adds parameters or initialization draws.** At its default each draws no random
  numbers, so `c0` (`vcpho5qm`) is their control. The only training-path change since `c0` is
  this one, default-off; the other uncommitted source edits are inference-only (the export
  filter and a new peak-aggregation helper).
- **Not built:** the per-molecule draw, the gated global message and DropKey (decision B's other
  options).

**Validation.**

- 14 new tests:
  - 8 in `tests/test_gine_gsa.py`: nothing is drawn when off; training-only with an unchanged
    state dict; substitution swaps whole messages for the detached batch mean; the noise scale
    matches each feature's batch standard deviation and a constant feature gets none; invalid
    settings are rejected.
  - 6 in `tests/test_masster_memorization_route.py`: only the shared encoder carries the keys;
    at eval each arm equals its control bit for bit, with identical initialization, and differs
    in training; the keys are rejected without `GINE_GSA`.
- The full suite passes: 4,390 passed, 1 skipped, 2 xfailed. Ruff is clean on the changed lines.
- **Resolved configs.** Through `load_config`, each arm differs from `c0` by its one key. Built
  through `MassterPL` on CPU, `c0`, `gm1` and `gm2` all have 9,212,861 parameters, with the
  setting on the shared `GINE_GSA` encoder and no separate proposal encoder.
- **Smokes.** Two epochs each on `debug_tiny_masster_cut_features.yml` (a `GINE_GSA`
  encoder), control and both keys by override, on the local RTX 2070 SUPER. All exited 0 at 340K
  parameters, and the train loss moved off the control's (−0.7445 and −0.7523 against −0.7544).
  The two-epoch val cosines are noise and are not read.

**Generate on Fir:** `scripts/manual_slurm/fir/submit_masster_mh_ft_memorization_route.sh gm`
(`fir:h100-3g.40gb`, 5 h, 64 epochs, seed 397623, no EMA). Read exactly as `k1`: the 54-63 tail
against `c0` (+0.003 to pass), `wrecall` within 0.005, then the offline gates on the checkpoint.
A pass stacks with EMA and with `k1` next.

## Global-message swap arms `gm3`-`gm6` (built 2026-09-25, not submitted)

**Why.** The user's framing: MASSter overfits because its embedding identifies the molecule, so
the fix is a feature-level CutMix. Standard CutMix mixes the labels too, and a MASSter target
(an allocation over one molecule's own candidate set) has no natural mix
([literature note](../../notes/memorization_literature_2026_09_24.md)). So only the message
that carries identity is swapped, and the label stays the molecule's own: swap noise. Where to
swap follows the route map. The global-attention message writes the key, and its inference-time
swap (`gsa_xmol`) had the largest train-specific damage of any condition (train −0.27, val
−0.12, gap 0.18 → 0.02-0.03). Atom inputs and the fragment/complement pools are generalizing
routes (about 1.2x), and swapping them would change the chemistry the label describes.

**Why a swap, next to `gm1`.** `gm1` substitutes the batch mean: one fixed vector the head can
learn to recognise and route around while it keeps using the key on the other atoms. A swapped
message is a real message, so the head cannot tell when the key is wrong, and relying on it
costs loss whenever a swap fires.

| arm | keys against `c0` | what one draw replaces in training |
| --- | --- | --- |
| `gm3_swap_atom` | `masster_global_message_swap: 0.3` (unit `atom`, the default) | an atom's whole message, from a random atom in the batch; `gm1`'s rate and draw, so `gm3` against `gm1` isolates an empty message against a wrong identity |
| `gm4_swap_molecule` | swap 0.3, `masster_global_message_swap_unit: molecule` | every atom of a molecule, each from a random atom of one other molecule (never itself): a coherent wrong identity, the per-molecule form decision B described |
| `gm5_swap_feature` | swap 0.3, unit `feature` | one (atom, feature) entry, from the same feature of a random batch atom (tabular-DAE swap noise) |
| `gm6_swap_molecule_p06` | swap 0.6, unit `molecule` | as `gm4` at twice the rate: the dose arm on the strongest unit |

**Design.**

- **Where.** `GINE_GSA._swap_global_message`
  ([`nn_blocks.py`](../../../src/fragnnet/model/nn_blocks.py)), called from
  `_perturb_global_message` after substitution and before Gaussian noise, so at the same place
  as `gm1`/`gm2`: after the message's norm, per encoder layer, training only.
- **Donors are detached**, so a swapped entry gets no gradient, as `gm1`'s mean.
- **The `atom` and `feature` donors are drawn over the whole batch**, so a donor falls in the
  atom's own molecule with probability (atoms in the molecule) / (atoms in the batch), under 1%
  at batch 512. The `molecule` unit excludes the molecule itself; a batch of one molecule is
  returned unchanged.
- **Same guarantees as `gm1`/`gm2`.** No parameters and no initialization draws; at swap 0 it
  draws nothing whatever the unit, so `c0` (`vcpho5qm`) is the control. Only the shared encoder
  gets the keys; the swap with a non-`GINE_GSA` encoder raises.
- **Four arms against a 0.0022 same-recipe spread.** One may clear +0.003 by chance; replicate
  any pass before promotion, as for `k1`.
- **Risk.** `r2` (no global attention) held val (+0.0006) without narrowing the gap, so the key
  may re-form in the 2-hop local embeddings. Gate 3, the route map on the arm's checkpoint, is
  what catches that. The same risk applies to `gm1`/`gm2`.
- **Not built:** a label-mixing CutMix. Its only form here trains on two molecules joined into
  one disconnected graph against their summed spectrum. That changes candidate enumeration, the
  precursor and the targets.

**Validation.**

- New tests in `tests/test_gine_gsa.py`: the atom swap takes whole detached messages from batch
  atoms, at the drawn rate, with no gradient to swapped atoms; the molecule swap moves every atom
  of a molecule to one other donor molecule, or none, and leaves a single molecule unchanged;
  the feature swap takes each entry from its own feature column, independently per entry. The
  existing tests now also cover the three units: nothing drawn when off, training only with an
  unchanged state dict, and invalid settings rejected.
- `tests/test_masster_memorization_route.py` runs its three `gm` model tests on the three units:
  only the shared encoder carries the keys, each arm equals its control at eval, and the swap is
  rejected without `GINE_GSA`.
- **Resolved configs.** Through `load_config`, `gm3` differs from `c0` by the swap key alone and
  `gm4`-`gm6` by the swap key and unit. Built through `MassterPL` on CPU, all have 9,212,861
  parameters, with the setting on the shared `GINE_GSA` encoder and no separate proposal encoder.
- **Smokes.** Two epochs each on `debug_tiny_masster_cut_features.yml`, control and the three
  units at 0.3 by override, on the local RTX 2070 SUPER. All exited 0 at 340K parameters. The
  epoch-1 train loss is −0.7544 for the control (as for `gm1`/`gm2`) and −0.7482 (atom), −0.7454
  (molecule) and −0.7683 (feature) for the arms. The two-epoch numbers are not read.
- The submitter's config checks pass for `swap` and `batch2`; locally it stops at the missing
  `pyteomics` wheel before generating anything, as before.
- Full suite: 4,414 passed, 1 skipped, 2 xfailed (20 new test cases over the 4,394 after the
  rename). Ruff is clean on the changed Python files.

**Generate on Fir:** `scripts/manual_slurm/fir/submit_masster_mh_ft_memorization_route.sh swap`
(`fir:h100-3g.40gb`, 5 h, 64 epochs, seed 397623, no EMA), or with the rest of batch 2. Read as
`gm1`: the 54-63 tail against `c0` (+0.003 to pass), `wrecall` within 0.005, then the offline
gates on the checkpoint. Read `gm3` against `gm1` as well as against `c0`.

## Batch 2 configs (2026-09-25, not submitted)

**Update 2026-09-25, 20:30 UTC: the original six jobs are running** (created 19:02-19:15 UTC,
same group): `gm1` `8ifz8p3t`, `gm2` `msmnydk7`, `k1_ema` `j4u25qfd`, `c0_s2` `1oxhvk0n`,
`k1_s2` `e39ko0zr`, `u2_s2` `qwhsp2fg`. All run at 3.0 min/epoch, as `c0`. `gm3`-`gm6` exist
only in the uncommitted working tree and are not submitted.

**Update 2026-09-25, ~21:45 UTC: all ten are running.** `gm3` `aijxahxb`, `gm4` `ou0ajppm`,
`gm5` `fr3dcnm1` and `gm6` `7qw60mn0` were created 21:04-21:16 UTC and were at epochs 9-13; the
original six were at epochs 49-53. Their configs and code are committed (`54fa5f2f`).

Early read at epochs 23-27 of 64, not the gate. Each value is the mean `val_cos_sim_0.01` delta over
the last 10 common epochs. The finished arms' early windows tracked their tails: `k1` +0.0057 at
18-27 against +0.0074 at 54-63, and `u2` +0.0135 against +0.0116.

| arm | control | epochs | delta | above control |
| --- | --- | --- | ---: | --- |
| `gm1` | `c0` | 18-27 | +0.0064 | 27 of 28 epochs; narrowing, +0.0100 at 21 to +0.0025 at 27 |
| `gm2` | `c0` | 18-27 | −0.0030 | 2 of 28; widening, −0.0059 at 27 |
| `k1_ema` | `u2` | 17-26 | +0.0039 | flat at +0.004 from epoch 19 |
| `k1_s2` | `c0_s2` | 15-24 | +0.0051 | 18 of 25 (`k1`'s own early read: +0.0055) |
| `u2_s2` | `c0_s2` | 14-23 | +0.0122 | 24 of 24 (`u2`'s own early read: +0.0139) |
| `c0_s2` | `c0` | 16-25 | +0.0011 | seed spread; per epoch −0.0019 to +0.0036 |

`val_candidate_wrecall` moves by at most 0.0008 in any pair.

Ten jobs: the six `gm` arms above, a stacking arm and a seed-811237 replicate of both passing
levers. All are unmerged `[M+H]+`/FT runs at 64 epochs on `fir:h100-3g.40gb`, in the same W&B
group. Generate on Fir with
`scripts/manual_slurm/fir/submit_masster_mh_ft_memorization_route.sh batch2`.

| config | against its control | control | checkpoints |
| --- | --- | --- | --- |
| `gm1_message_substitution.yml` | `masster_global_message_substitution: 0.3` | `c0` (`vcpho5qm`) | uploaded |
| `gm2_message_noise.yml` | `masster_global_message_noise_std: 0.5` | `c0` | uploaded |
| `gm3_swap_atom.yml` | `masster_global_message_swap: 0.3` | `c0` | uploaded |
| `gm4_swap_molecule.yml` | swap 0.3, unit `molecule` | `c0` | uploaded |
| `gm5_swap_feature.yml` | swap 0.3, unit `feature` | `c0` | uploaded |
| `gm6_swap_molecule_p06.yml` | swap 0.6, unit `molecule` | `c0` | uploaded |
| `k1_ema.yml` | the two `k1` keys plus coverage monitoring, against `u2` | `u2_ema999` (`c8hh9j1b`) | uploaded |
| `c0_s2_control.yml` | `seed: 811237` against `c0` | — | not uploaded |
| `k1_s2.yml` | the two `k1` keys | `c0_s2` | not uploaded |
| `u2_s2_ema999.yml` | `ema_decay: 0.999` | `c0_s2` | not uploaded |

- **Stacking (`k1_ema`).** It passes if it beats `u2` by at least +0.003. `k1` removed the late
  rise in val loss, which EMA keeps, so the two may add.
- **Replicates.** They use seed 811237, the seed `c4a_base_s2` used, because `k1` and EMA are
  each n=1 and the only same-recipe pair here differs by 0.0022. Each replicate is read against
  `c0_s2` at the same gate. `c0_s2` against `c0` also gives a second estimate of the seed spread.
- **No checkpoints for the replicates.** They are read on W&B only; the offline gates run on the
  seed-397623 checkpoints. Checkpointing stays on, so a requeue still resumes, and the files stay
  on Fir. The submitter requires `upload_checkpoints: false` for these three and `true` for the
  rest.
- **Checks.** Each resolved config differs from its control by exactly the keys in the table,
  plus `seed` and `upload_checkpoints` against the seed-397623 base. The submitter's checks pass
  for all ten (the four swap arms added the same day); the local run stops at the missing
  `pyteomics` wheel before generating anything.
  A two-epoch smoke of `k1` + EMA, the one combination not run before, exits 0 on the debug
  config.

### Batch 2 tail read, six finished runs (2026-09-26)

Read from W&B at 00:20 UTC. `gm3`-`gm6` were still running (epochs 57-61). Tail = mean
`val_cos_sim_0.01` over epochs 54-63; coverage = the same mean of `val_candidate_wrecall`.

| arm | run | tail | control | delta | above control | coverage delta |
| --- | --- | ---: | --- | ---: | --- | ---: |
| `k1_ema` | `j4u25qfd` | 0.7421 | `u2` (0.7392) | +0.0029 | 51 of 64 epochs | not logged by `u2`; +0.0007 vs `c0` |
| `gm1` | `8ifz8p3t` | 0.7327 | `c0` (0.7276) | +0.0051 | 63 of 64 | +0.0003 |
| `gm2` | `msmnydk7` | 0.7247 | `c0` | −0.0029 | 6 of 64 | −0.0006 |
| `k1_s2` | `e39ko0zr` | 0.7366 | `c0_s2` (0.7318) | +0.0048 | 57 of 64 | +0.0001 |
| `u2_s2` | `qwhsp2fg` | 0.7390 | `c0_s2` | +0.0072 | 64 of 64 | +0.0004 |
| `c0_s2` | `1oxhvk0n` | 0.7318 | `c0` | +0.0042 | 56 of 64 | +0.0002 |

- **The seed spread is 0.0042.** `c0_s2` differs from `c0` by `seed` alone, so this is the
  cleanest floor measured on this lineage, about twice the 0.0022 `c0`/`f0` pair used so far and
  above the +0.003 gate. A single-seed delta under about +0.005 is not resolved.
- **Both passing levers replicate.** `k1` is +0.0074 at seed 397623 and +0.0048 at 811237; EMA
  +0.0116 and +0.0072. EMA is the larger on both seeds.
- **Stacking is not shown.** `k1_ema` has the best tail on this lineage (+0.0145 over `c0`), but
  it is +0.0029 over `u2`, under the gate and inside the seed spread.
- **`gm1` passes gate 1 on one seed** (+0.0051); **`gm2` fails** (−0.0029).
- Coverage moves by at most 0.0007 anywhere, so none of this is candidate-set coverage.
- Not yet run: gates 2-4 (ranking decomposition, route map, novelty bins) on the new
  checkpoints.

**Update 01:35 UTC: `gm3`-`gm6` finished.** Same tail and control (`c0`, 0.7276); `gm1` (0.7327)
is the batch-mean form of the same message regularizer.

| arm | run | tail | delta vs `c0` | above `c0` | vs `gm1` | coverage delta |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `gm3` swap per atom, 0.3 | `aijxahxb` | 0.7300 | +0.0023 | 53 of 64 | −0.0028 | +0.0002 |
| `gm4` swap per molecule, 0.3 | `ou0ajppm` | 0.7282 | +0.0006 | 56 of 64 | −0.0045 | −0.0001 |
| `gm5` swap per feature, 0.3 | `fr3dcnm1` | 0.7248 | −0.0029 | 2 of 64 | −0.0080 | −0.0007 |
| `gm6` swap per molecule, 0.6 | `7qw60mn0` | 0.7251 | −0.0025 | 11 of 64 | −0.0076 | −0.0005 |

- **No swap arm passes gate 1**, and every one is below `gm1`. Swapping in another molecule's
  real message does worse than the batch mean, per atom as well as per molecule. Per-feature
  swaps and the 0.6 rate hurt.
- **The swap family is closed.** `gm1`, batch-mean substitution, is the only global-message
  arm that passes, on one seed and within about 0.001 of the seed spread.
  - This is a resource decision, not a measured harm. Each swap delta is a single seed inside
    the 0.0042 spread.
  - The ordering is solid, though: on the same seed every swap arm sits below `gm1`, and the 0.6
    rate is below 0.3.

## Learning-rate-schedule arms `s1` and `s1_k1_ema` (2026-09-25)

**Built, not submitted.** *Finished 2026-09-28 (`nq2al58c`, `f2oacs52`):
[Batch 4 read](#batch-4-read-from-wb-2026-09-28).* Plan decision C, config only. No code changed: the schedule path is
`build_lr_scheduler` in `SpectrumPL.configure_optimizers`, which `MassterPL` inherits, and it ran
on the merged lineage as `d3`.

| config | against its control | control | checkpoints |
| --- | --- | --- | --- |
| `s1_lr_cosine.yml` | the four schedule keys | `c0` (`vcpho5qm`, tail 0.7276) | uploaded |
| `s1_k1_ema_lr_cosine.yml` | the four schedule keys | `k1_ema` (`j4u25qfd`, running) | uploaded |

- **The schedule is `d3`'s.** `lr_schedule: true`, `lr_schedule_type: cosine`,
  `lr_min_ratio: 0.01`, `lr_total_steps: 19712`, and the template's `lr_warmup_steps: 1000`.
  - `c0` takes 308 optimizer steps per epoch (W&B `trainer/global_step` rises by 308 each epoch
    and ends at 19,711), so 64 epochs are 19,712 steps.
  - The rate rises to 1e-3 at step 1000 (3.2 epochs), is 5.5e-4 at epoch 32, 7.5% of peak where
    the 54-63 tail starts, and 1e-5 at the last step. Averaged over the run it is half of `c0`'s.
  - The warmup stays at 1000 steps, not `d3`'s share of the run (11 of 128 merged epochs): it
    is there for the post-LN stack, whose early instability is counted in optimizer steps.
- **Reading.**
  - `s1` against `c0`: +0.003 on the 54-63 tail, with `val_candidate_wrecall` within 0.005.
    That tail compares settled weights (7.5-1% of peak) with constant-rate ones, which is the
    contrast decision C asks for.
  - `s1` against `u2` (`c8hh9j1b`, 0.7392) as well. EMA and decay both remove late-training
    weight noise, so a gain near `u2`'s may be the same gain.
  - The eval-mode train/val gap on `s1`'s best checkpoint, as in the
    [Step 1 table](#eval-mode-trainval-gap-on-the-step-1-checkpoints-2026-09-25). The schedule
    is a memorization lever only if the gap narrows; train and val rising together is an
    optimization gain, as EMA's was.
  - `s1_k1_ema` against `k1_ema`, same gate: whether the schedule adds to the two levers that
    passed. Read it only if `k1_ema` passes against `u2`. It narrowly did not (+0.0029,
    [read](#batch-2-tail-read-six-finished-runs-2026-09-26)), so `s1_k1_ema` is optional.
  - **The schedule also halves the weight decay.** AdamW shrinks each weight by `1 − lr·wd` per
    step at the current rate (torch 2.4.1), so both arms decay half as much over the run as their
    controls (an undriven weight ends at 0.905 against 0.821) and almost not at all in the tail.
    Each arm is LR decay plus half the weight decay, not LR decay alone. At `wd` 0.01 the decay
    timescale, `1/(lr·wd)` = 100,000 steps (325 epochs), is five times this run, so the second
    change is small.
- **Checks.**
  - Through `load_config`, each config differs from its control by exactly `lr_schedule`,
    `lr_schedule_type`, `lr_total_steps` and `lr_min_ratio`, plus W&B naming.
  - The rates above come from `build_lr_scheduler` on the resolved `s1` config.
  - Two-epoch smokes on `debug_tiny_masster_cut_features.yml`, local GPU, both exit 0: the
    schedule alone, and the schedule with the two `k1` keys and `ema_decay: 0.9`. In the second,
    `last.ckpt` holds the scheduler at step 32, equal to `global_step`, and the optimizer's rate
    (1.258e-4) is the cosine value at that step exactly. The scheduler steps once per optimizer
    step through the MASSter path, with the EMA callback alongside.
  - `bash -n` passes on the generator. Its checks pass for `lr`, `s1` and `s1_k1_ema`; the local
    run stops at the missing `pyteomics` wheel before generating anything, as for batch 2.
- **Generate** on Fir with `submit_masster_mh_ft_memorization_route.sh lr` (both), or `s1` or
  `s1_k1_ema` alone. The schedule does not change the compute, so the 5 h box holds.
- **Name.** `s1` is decision C's name. The Step 3 option also called `s1` (set dropout) is not
  built; give it another name if it ever is.

## Weight-decay arm `w1` (2026-09-25)

**Finished null: +0.0010 over `c0` (`tggzldtj`,
[read](#batch-3-read-from-wb-2026-09-26)).** Config only. `w1_wd10.yml` is `c0` with `weight_decay: 0.1` instead of
0.01, read against `c0` (`vcpho5qm`, tail 0.7276). Checkpoints are uploaded.

- **Why.** AdamW shrinks each weight by `1 − lr·wd` per step, so the decay timescale is
  `1/(lr·wd)` steps. At `c0`'s 0.01 that is 100,000 steps (325 epochs), five times the 64-epoch
  screen, and an undriven weight ends the run at 0.82 of its value. At 0.1 it is 10,000 steps
  (32 epochs), and the weight ends at 0.14. Weight decay has never been varied on this lineage.
- **Prior evidence, merged only.**
  - `wd10` (`kalc4mzo`, 0.1) was the one arm of the 2026-09-19 screen to narrow the train/val
    gap (0.122 against 0.128-0.130) without lowering train cosine, at +0.0015 val inside a
    ±0.007 screen. It ran the pre-2026-09-20 architecture and crashed at epoch 65 of 630.
  - The 128-epoch rerun `c4d_wd10` was prepared on 2026-09-22 and never submitted; W&B has no
    such run.
  - `b3_wd_groups` (norms and biases exempt, at 0.01) was an exact null, and `p3` (0.1 on the
    proposal branch only) a null.
- **Scope.** The decay reaches every tensor, as in `c0` and the merged arm
  (`weight_decay_exclude_norm_bias: false`), so norm gains, biases and the inducing tokens decay
  at 0.1 too. If `w1` underfits, the follow-up is the same decay with those exempt.
- **Reading.**
  - Against `c0`: +0.003 on the 54-63 tail, with `val_candidate_wrecall` within 0.005.
  - The eval-mode train/val gap on the best checkpoint, as in the
    [Step 1 table](#eval-mode-trainval-gap-on-the-step-1-checkpoints-2026-09-25). The question is
    whether the gap narrows with val up. Train down with val flat or down is underfitting.
  - The seed spread is 0.0042 ([batch 2 read](#batch-2-tail-read-six-finished-runs-2026-09-26)),
    so a result between +0.003 and about +0.005 needs a second seed before it counts.
- **Checks.**
  - Through `load_config`, `w1` differs from `c0` by `weight_decay` alone, plus W&B naming.
  - A two-epoch smoke on `debug_tiny_masster_cut_features.yml` with `weight_decay=0.1`, local
    GPU, exits 0. Its `last.ckpt` holds one AdamW group over all 156 parameter tensors, at
    `weight_decay` 0.1 and `lr` 1e-3.
  - `bash -n` passes. The generator's checks pass for `w1`; the local run stops at the missing
    `pyteomics` wheel before generating anything.
- **Generate** on Fir with `submit_masster_mh_ft_memorization_route.sh w1`.

## Energy vs parts read (2026-09-26)

Inference only, on f0 (`3bej6fq7`), with the probe's new `ctx` scope and a per-peak rarity
analysis ([finding](../../findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md)).
The lookup is per molecule: not per energy (±1 NCE costs train and val alike) and not per part
(train fit is flat across molecule novelty and nearly flat across break rarity). Val transfer
runs through familiar breaks. This confirms the plan's target, molecule identity on every
route, and adds break-local evidence as the signal that transfers.

## Light-side token arms `nl1`-`nl3` (built 2026-09-26, not submitted)

**Submitted 2026-09-26; final read [below](#final-read-of-nl1-and-nl2-2026-09-27).**

- `nl1` finished at +0.0037, an unresolved single-seed pass.
- `nl2` is −0.0021 against `r2`, a mild underfit.
- `nl3` underfits and crashed at epoch 30.

The mid-run projection of two nulls is superseded.

The [side-of-cut probe](../../findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md#side-of-cut-probe-2026-09-26)
found the heavier side of each cut the more memorizing input: removing its identity costs train
1.78x what it costs val (2.8x at NCE ≤ 20), and val barely uses it. The lighter side carries 2.5
times more of what val uses. So each candidate is represented by its lighter side: the fragment
itself, or the neutral loss of a large fragment.

- **Option.** `masster_light_side_token: true` puts the lighter side's pool (by neutral mass,
  the probe's rule) in the token's fragment slot and zeroes the complement slot. The heavier side
  enters only through the new descriptor `c_heavy_side`: a lighter-side flag and the heavier side's
  `log1p` element counts (19 columns). The candidates, proposal, boundary block and allocator are
  unchanged. It requires unique-fragment mode and `c_heavy_side`; off by default.
- **Code.** `CandidateFeatureInputs.fragment_is_lighter_side` and `component_element_counts`, and
  the `c_heavy_side` spec, appended last in canonical order so no existing token changes
  (`candidate_features.py`). The substitution is in `MassterModel`'s unique-fragment branch, and
  the key is wired through `masster_pl.py` and declared in `config/template.yml`.

| arm | config | change | controls |
| --- | --- | --- | --- |
| `nl1` | `nl1_light_side.yml` | `c0` + light-side token + `c_heavy_side` | `c0` (`vcpho5qm`) |
| `nl2` | `nl2_light_side_gine.yml` | `nl1` + `mol_gnn_type: GINE` | `c0`, `r2` (`9lt5s2u8`) |
| `nl3` | `nl3_light_side_gine_k1.yml` | `nl2` + `k1`'s dropout (0.2, 0.2) | `nl2`, `k1` (`wg0nxymi`) |

`nl2` leaves the token with no whole-molecule representation. The lighter-side pool and the
boundary block come from 2-layer local embeddings, and the heavier side is element counts. `r2`
alone held val while the key re-formed through local message passing, and the heavier side's
pool is its likely carrier. The remaining route is the candidate set, through allocator
attention.

- **Checks.**
  - Three new tests:
    - `c_heavy_side` values on ethanol (the precursor, a lighter CH3 and a heavier C2H5);
    - the token receives exactly the lighter side's pool with the complement slot zeroed, the
      precursor's slot zero, and the same candidates;
    - both validation errors.
  - The descriptor-width test is updated. 231 pass across the MASSter model, candidate-feature
    and probe tests, and ruff is clean.
  - Through `load_config`, `nl1` differs from `c0`, and `nl2` from `r2`, by
    `masster_candidate_features` (+`c_heavy_side`) and `masster_light_side_token` alone. `nl3`
    differs from `nl2` by the two `k1` keys.
  - Production parameters: `c0` 9,212,861, `nl1` 9,217,725 (+19 × 256), `nl2`/`nl3` 8,033,469.
  - Two-epoch smokes on `debug_tiny_masster_cut_features.yml`, local GPU, both exit 0:
    - `nl1`'s options, 341K parameters against the debug control's 340K;
    - `nl3`'s, 308K against `r2`'s 307K;
    - the resolved token reads `c_heavy_side:19` (44 columns).
  - The old f0 checkpoint still loads with `strict=True`.
  - `bash -n` passes on the submit script, and the three configs pass its checks.
- **Initialization.** The token projection is 19 columns wider, so initialization differs from
  `c0` from that layer on. Read against the seed spread (0.0042), not the replicate floor.
- **Reading.** As for the other arms: the tail mean against the control and `wrecall`. Offline,
  on the best checkpoint:
  - the route-map probe: `cc` and `boundary` on every arm, `gsa` on `nl1` only. The `side`
    scope does not apply, because its mass-fraction rule assumes both pools are in the token;
  - the parts test (`output/memorization_parts_ev_2026_09_26/`);
  - Tanimoto bins.

  A pass narrows the eval-mode gap well beyond `k1`'s 9% with val at least at the control, and
  train's rare-minus-common break contrast rises toward val's.
- **Generate** on Fir with `submit_masster_mh_ft_memorization_route.sh nl` (all three), or `nl1`,
  `nl2` or `nl3` alone.

## Batch 3 read from W&B (2026-09-26)

Read at 03:45 UTC on 2026-09-27. Tail and gate as before: mean `val_cos_sim_0.01` over epochs
54-63, seed spread 0.0042. Arms still running are compared at the latest ten matched epochs, then
projected to the tail with a line fitted over the 21 finished arms of this recipe (this group plus
`u2`, `u3`, `h1`, `h2` and `f0`). Each arm's gap at the same window is regressed on its tail gap.
Loss deltas are train-mode (train) and eval-mode (val) primary loss. They are compared only
between runs with identical loss settings, which holds within this group.

| arm | run | state | window | cosine | vs `c0` | projected tail vs `c0` | train loss vs `c0` | val loss vs `c0` |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `w1` | `tggzldtj` | finished | 54-63 | 0.7286 | +0.0010 | — | +0.037 | −0.016 |
| `nl1` | `107bmt51` | epoch 28 | 19-28 | 0.7240 | +0.0015 | +0.0015 | −0.009 | −0.001 |
| `nl2` | `djhbz6z7` | epoch 29 | 20-29 | 0.7230 | +0.0002 | +0.0002 | +0.003 | +0.004 |
| `nl3` | `kkp8ug11` | epoch 26 | 17-26 | 0.7045 | −0.0170 | −0.0146 | +0.142 | +0.033 |

The fit at these windows has r = 0.96-0.98 and a residual SD of 0.0018-0.0021. For scale:
- the seed pair (`c0_s2` − `c0`) moves the tail losses by +0.002 train and −0.024 val;
- `k1` by +0.080 and −0.040;
- `u2` by −0.002 and −0.042.

- **`w1` is null.** +0.0010, above `c0` at 27 of 64 epochs, coverage equal. Ten times the decay
  lowers train fit by half as much as `k1` does, and its −0.016 val loss is inside the seed pair's
  −0.024. Val held, so this is not the underfitting that the norm/bias-exempt follow-up was for.
  Weight decay is closed on this lineage.
- **`nl1` and `nl2` look null mid-run, and train fit does not move.** With the heavier side's
  pool gone from the token, train loss stays where it was (−0.009 and +0.003). Yet the
  [side-of-cut probe](../../findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md#side-of-cut-probe-2026-09-26)
  rated that pool the most train-specific input. `nl2` − `r2` is −0.0011 at epochs 20-29.
  **Superseded by the [final read](#final-read-of-nl1-and-nl2-2026-09-27):** `nl1` ends at
  +0.0037, `nl2` underfits against `r2`, and neither arm can show rerouting.
- **`nl3` fails by underfitting.** `k1`'s two dropouts gain +0.0057 on `c0` at this window, with
  val loss down (train +0.109, val −0.024). On `nl2` the same keys cost −0.017, with train loss
  +0.142 and val loss up +0.033. `nl3` is below `nl2` at every epoch from 0, by 0.009-0.037.
  Without the heavier-side pool or global attention, the token and the boundary block are the
  head's remaining structure inputs, and 20% dropout on each starves it. That suggests `k1`
  works through redundancy between routes, and does not stack with a change that removes
  routes.
- `nl2` and `nl3` take 176 s per epoch against `c0`'s 182 s, from dropping global attention.
- **`u3`** is read in the
  [ranking-transfer plan](masster_ranking_transfer_plan.md#switch-ema-arm-u3-built-2026-09-25-not-submitted):
  switch EMA fails against `u2`.
- **`n1`-`n3`** are read in the
  [QET finding](../../findings/masster/masster_qet_projection_candidate_union_2026_09_26.md#validation-arm-n1).
- **`sp0` (`85mfkk8i`) and `sp1` (`bu6dzjph`)** started at 03:43 UTC and had no epoch logged.
  They then collapsed on the broken CE-context sampler, crashed at epoch 2, and were resumed
  from their epoch-2 checkpoints after the fix. They are read in the
  [shared-proposal audit](../../findings/masster/masster_shared_proposal_pathway_audit_2026_09_26.md).

**Update, about two hours later.** `nl3` crashed at epoch 30 (cause not checked). It was
−0.018 against `nl2` over epochs 21-30, so it needs no resubmission. `nl1` at epoch 39 and `nl2`
at 41 stay null: +0.0022 and −0.0013 against `c0` over their last ten epochs. `nl1`'s train loss
equals `c0`'s (−0.0006).

**Next.** Read `nl1` and `nl2` at epoch 63, then run the route map (`cc`, `boundary`; `gsa` on
`nl1`) on their checkpoints to find the route that replaced the heavier-side pool. `nl3` needs
no offline read.

### Final read of `nl1` and `nl2` (2026-09-27)

Both finished 64 epochs. Tail = mean of `val_cos_sim_0.01` over epochs 54-63. The losses are the
same-window deltas of train-mode and eval-mode primary loss.

| arm | run | tail | vs `c0` | vs `r2` | above `c0` | train loss | val loss |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `nl1` | `107bmt51` | 0.7313 | **+0.0037** | — | 52 of 64 | +0.0039 vs `c0` | −0.0041 vs `c0` |
| `nl2` | `djhbz6z7` | 0.7261 | −0.0015 | **−0.0021** | 25 of 64 | +0.0095 vs `r2` | +0.0221 vs `r2` |

- **`nl1` is an unresolved pass, not a null.** It clears the +0.003 gate, but the token is 19
  columns wider, so initialization differs from `c0`, and the delta is inside the 0.0042 seed
  spread. It needs a second seed. Coverage is equal (0.9396). Its gain rose through the run:
  +0.0015 at epochs 19-28, +0.0022 at 30-39, +0.0037 at 54-63.
- **`nl2` underfits against its same-code control `r2`.** Train and val loss are both worse and
  val cosine is lower. Train fit moved after all.
- **Neither arm can show rerouting.** Both keep the boundary block (bit-identical under the
  toggle), `c_mass`, the new `c_heavy_side` (close to the heavier side's formula) and the full
  candidate set the allocator attends over. So "the lookup moved to another route" cannot be told
  apart from "the heavier side's identity never left the token".
- **`nl1`'s gain needs global attention.** The same token change without it (`nl2`) is below
  `r2`.
- **The route map on both checkpoints has run (2026-09-27,
  [read](../../findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md#set-message-probe-on-c0-r2-nl1-and-nl2-2026-09-27)).**
  - `nl1` still writes the key in global attention (2.89x, excess 0.127, as on `c0`).
  - In both arms the allocator's set message carries more of train and val fit than on `c0`
    (val drop 0.09 against 0.02).
  - So the heavier side's removal left the encoder key in place, and moved molecule context
    into the set message.
- **Next:** a second seed of `nl1` against `c0_s2`. The
  [audit](../../findings/masster/masster_memorization_attempts_audit_2026_09_27.md) has the
  noise floor.

## Sink-dropout arm `m2` (built 2026-09-26, not submitted)

**Submitted 2026-09-27 (`41mnwm9f`, created 04:53 UTC); running.** At epochs 31-40 it is +0.0035
over `c0` and +0.0238 over `m1`, so it tracks `c0`, not `m1`. Read the 54-63 tail when it
finishes.

`m1`'s sink applied to a random half of the training spectra: plan item 1 of
[Next, ranked by evidence](#change-list-2026-09-24), without the variational shrinkage.

- **Why.** `m1` (`liwjgdq6`) failed at −0.0203 because the sink took transferable fit along with
  the lookup ([read](#step-1-tail-read-from-wb-2026-09-24)). Once the sink fits a spectrum, the
  shared weights get almost no gradient from it, and `m1` applied it to every exposure from
  epoch 4. MemSinks mixes shared and specific units, so the shared path is also trained without
  the specific ones. In `m2` the shared path must fit half of each molecule's exposures alone.
- **Option.** New key `masster_memorization_sink_dropout` (default 0, which is `m1`): the
  training-only probability, drawn per spectrum at each step, that the sink row is zeroed. A zero
  row adds nothing, so a dropped spectrum runs the shared model exactly. Kept rows are not
  rescaled: the sink is on or off. It requires `masster_memorization_sink_dim > 0` and adds no
  parameters. Code: `MassterModel(memorization_sink_dropout=...)` in `masster_model.py`, wired
  through `masster_pl.py` and declared in `config/template.yml`.
- **Config.** `m2_sink_dropout.yml` is `m1_sink.yml` plus the key at 0.5: 16-dim rows, start
  epoch 4, never used at evaluation. Controls: `c0` (`vcpho5qm`, tail 0.7276) and `m1` (tail
  0.7074).
- **Reading.** The tail against `c0` needs +0.003, and a replicate if under about +0.005. Against
  `m1` it shows how much of `m1`'s loss the dropout recovers. Also read coverage within 0.005 of
  `c0` and the eval-mode train/val gap on the best checkpoint. It is a long shot: it must recover
  0.023 over `m1`, and the only positive evidence is at LLM scale.
- **Checks.**
  - 4 new tests in `tests/test_masster_memorization_route.py`:
    - no new parameters, and a state dict equal to `m1`'s;
    - each spectrum matches exactly the sink-on or the shared output, with all four keep
      patterns across 24 seeds, so spectra are drawn independently;
    - no dropping in eval mode;
    - invalid settings rejected.
  - Each test catches a matching broken variant: one mask per batch, or dropping in eval mode.
  - 323 pass across the memorization-route, MASSter model and config test files. Ruff is clean.
  - Through `load_config`, `m2` differs from `m1` by the key and W&B naming only.
  - Two 2-epoch smokes on a 512-spectrum subsample exit 0, with the start moved to epoch 0 so
    the sink runs. At 0.5, 390 sink rows moved, and at 0 all 498 train molecules' rows did.
    Val cosine at 0.5: 0.075 → 0.230.
  - The submitter's config checks pass. Locally it stops at the missing `pyteomics` wheel before
    generating anything.
- **Generate** on Fir with `submit_masster_mh_ft_memorization_route.sh m2`, after committing.

## Encoder atom-mask and bond-dropout arms `am1` and `de1` (built 2026-09-27, not submitted)

**Why.** The user proposed atom drop and bond drop in the GNN, and a virtual node, on 2026-09-27.
The [audit](../../findings/masster/masster_memorization_attempts_audit_2026_09_27.md) found that
removing an identity route never helped, while noise on a route that other routes back up did:
`k1` (replicated) and `gm1` (one seed). Every earlier atom dropout acted after the encoder, on the
fragment pools (`e2`/`e3`, null), or everywhere at once (`drop03`, −0.0149). These two arms
perturb the molecule the encoder sees on each exposure and leave every route in place.

- DropEdge was under "Not changing" because the route map put the key in global attention, not
  topology. The audit found the route map does not predict which arms pay (`r2` null; `k1`
  predicted weak, then replicated), so that reason no longer holds.
- **Virtual node: not built.** `GINE_GSA` already runs full per-molecule softmax attention in
  every layer. A virtual node is that attention with uniform weights, so it adds no path, and
  `r2` found removing global attention null. The one variant with a reason is a narrow noisy
  virtual node in place of global attention, the audit's bottleneck lever (rank 8). It waits for
  `gm1`'s replicate.

**Code.** Both keys act only on the shared precursor encoder, never on `proposal_encoder` or the
`r1` local encoder. At 0.0 neither draws random numbers, and the state dict is unchanged.

- `masster_encoder_atom_mask` → `GNN(atom_mask=...)` in `nn_blocks.py`. In training, per atom and
  forward pass, the projected input (after `input_project`) is replaced by one learned vector,
  `atom_mask_embedding`. The vector exists only when p > 0 and is zero-initialized, so every other
  parameter initializes as in the control. Any `gnn_type` takes it. With `context_in_gnn` it would
  mask the atom's context too; `c0` has that off.
- `masster_encoder_bond_dropout` → `GINE_GSA(bond_dropout=...)`. One draw per forward pass per
  bond, keyed by the unordered atom pair, so both directions drop together without assuming an
  edge order (one `th.unique` host sync per pass). A dropped bond leaves `GINEConv` in every layer.
  - Kept messages are scaled by w = 1/(1 − p) exactly: passing `(w·x, x)` and `w·e` gives
    relu(w·x_j + w·e) = w·relu(x_j + e) and leaves the root term unscaled.
  - Global attention, the candidate set, the cut-bond features and the boundary encoder still see
    every bond.
  - It requires `mol_gnn_type: GINE_GSA`, and `MassterModel` raises otherwise.

**Configs.** `am1_atom_mask.yml` and `de1_bond_drop.yml` are `c0` plus the key at 0.1. The low rate
is on purpose: noise on an already-stripped model underfits (`nl3`), and so did `drop03`. A rate
ladder follows only a pass. The control is `c0` (`vcpho5qm`, tail 0.7276). `am1` adds 384
parameters (the vector) and `de1` none.

**Checks.**

- 18 new test cases pass (15 functions):
  - `tests/test_gnn_atom_mask.py`:
    - off, it draws nothing and adds no parameter;
    - on, it adds only the zero vector, with every other initialization unchanged;
    - it is training-only;
    - it replaces whole rows (about p of them), and the vector gets gradient;
    - invalid rates are rejected.
  - `tests/test_gine_gsa.py`:
    - both directions of a bond drop together, with features kept aligned;
    - the (w·x, x) scaling identity holds against a hand-written `GINEConv` sum;
    - it is training-only and adds no parameters;
    - there is one draw per pass across both layers;
    - invalid rates are rejected.
  - `tests/test_masster_memorization_route.py`:
    - each key reaches only the shared encoder;
    - each arm equals its control at eval, with the same initialization;
    - bond dropout requires `GINE_GSA`;
    - the atom mask works on plain `GINE`.
- 93 pass across those three files, and the full suite passes (4,535; 1 skipped, 2 xfailed).
  Ruff is clean.
- Through `load_config`, each arm differs from `c0` by its key and W&B naming only.
- **Smokes.** Two epochs each on `debug_tiny_masster_cut_features.yml`, on the local RTX 2070
  SUPER: the control and both keys by override. All exit 0 at 340K parameters.
  - Epoch-1 train loss: `am1` −0.7624 and `de1` −0.7491, against −0.7544 for the control.
  - The control's −0.7544 equals the `gm` control smoke's, so the defaults change nothing.
- The configs pass the submitter's grep checks. Nothing was generated locally.

**Reading.** First, `val_candidate_wrecall` within 0.005 of `c0`: the proposal reads the same
encoder. Then the 54-63 tail against `c0`. By the audit's rule, a single seed needs about +0.006;
otherwise it gets a second seed at 811237 against `c0_s2`. Then gates 2-4 on the checkpoint. A
pass stacks with EMA (`u2`) next. The cost is expected to equal `c0`'s 3.0 min/epoch but is
unmeasured.

**Generate** on Fir with `submit_masster_mh_ft_memorization_route.sh drop` after committing.

### Local screen of `am1`/`de1` (2026-09-27)

A local screen on the RTX 2070 SUPER before Fir, at the user's request. It is not comparable to
Fir numbers: it uses a quarter of the molecules and batch 128. Artifacts (git-ignored):
`output/local_encoder_noise_2026_09_27/` holds the queue script, split, logs, `parse_logs.py`,
`eval_matched.py` and `eval_matched.json`, and the checkpoints are in
`ckpts/local_{control,am1,de1,control_s2}/`.

- **Setup.** A molecule-level subset of `nist20_inchikey`, seed 20260927:
  - 3,000 train molecules (37,111 spectra) and 1,000 val molecules (12,381).
  - `test_ids.csv` holds 1,000 of the *train* molecules (12,300 spectra), so a test pass is the
    eval-mode train fit.
  - Each run is its committed config plus `train_batch_size`/`eval_batch_size` 128 (7.4 GB peak;
    512 does not fit 8 GB), 24 epochs and `eval_test_split: true`.
  - Runs: `c0`, `am1`, `de1`, and `c0` at seed 811237 as the floor.
  - A run takes 57 min, at 2.3 min/epoch and 290 steps. All four exit 0.
- **The regime reproduces the problem.** The control's val loss bottoms at epoch 19 while train
  loss keeps falling, and its eval-mode gap is about 0.24.

| run | tail 19-23 | tail 12-17 | best (epoch) | epoch-23 train-eval | epoch-23 gap | wrecall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `c0` | 0.6287 | 0.6293 | 0.6331 (17) | 0.8710 | 0.2436 | 0.9296 |
| `c0` seed 811237 | 0.6303 | 0.6268 | 0.6361 (23) | 0.8846 | 0.2485 | 0.9281 |
| `am1` | **0.6362** | 0.6300 | 0.6399 (21) | **0.8650** | **0.2336** | 0.9283 |
| `de1` | 0.6242 | 0.6213 | 0.6295 (23) | 0.8788 | 0.2494 | 0.9281 |

Tail = mean val `cos_sim_0.01` over the epochs given. The epoch-23 columns score each run's
`last.ckpt` with `eval_matched.py`. The post-fit test pass scored the *best* checkpoints, whose
epochs differ (17-23), and the seed-811237 control shows that alone moves train fit by 0.02.
Re-scoring the control's best checkpoint reproduces its logged 0.6331 / 0.8647.

- **Floor (one pair).** The two controls differ by +0.0016 on the 19-23 tail, −0.0025 on 12-17,
  and 0.0049 in epoch-23 gap. Single epochs swing ±0.01.
- **`am1` is the one positive signal.**
  - Its 19-23 tail is +0.0067 over the control mean, and +0.0059 over the higher control.
    Tail 12-17 was even, so the gain arrives as the control starts to overfit.
  - It has the lowest epoch-23 train fit and the narrowest gap: −0.0125 against the control mean,
    and −0.010 against its same-seed control, twice the one-pair gap spread.
  - Val loss is lower by 0.03 late. Coverage is unchanged.
  - This is the pattern the plan's objective asks for: a narrower gap with val up, not
    underfitting. Every earlier arm that narrowed the gap lost val.
- **`de1` is null to negative.** Its 19-23 tail is −0.0053 against the control mean; it is below
  the control in all of epochs 12-17 and in 8 of 12 from epoch 12 on. Its gap (0.2494) is inside
  the control spread. On a 2-layer local encoder a dropped bond removes much of each nearby
  atom's 2-hop view, which fits the DropEdge caution in the literature note.
- **Caveats.** One seed per arm, one control pair and a 1,000-molecule val set; batch 128, not
  512; 24 epochs on a quarter of the molecules.
- **Next.** Submit `am1` on Fir; a local seed-811237 replicate against the second control would
  firm it up first (1 h). `de1` on Fir is optional: the local read predicts no pass.
- **Rate ladder: stopped after 0.3.** The run was `am1` with only `masster_encoder_atom_mask`
  overridden, on the same setup (`run_queue_am_ladder.sh`).
  - At 0.3 (`am_p03`), the 19-23 tail is 0.6242: −0.0053 against the control mean, and below the
    control at all 24 epochs. Best is 0.6287 at epoch 21.
  - Train-mode loss is 0.14 above the control, and the best checkpoint's train-eval is 0.8541, the
    lowest of any run. The gap narrows only because train fit falls: underfitting.
  - The deficit closes over the run (−0.02 early, −0.001 to −0.006 at epochs 20-23), so 64
    epochs might recover it; this screen cannot tell.
  - 0.5 was stopped 3 minutes in, because it would underfit further. 0.2 was stopped at epoch 4,
    after the user judged the ladder not useful. Neither was read; logs are kept as `*_stopped.log`.
  - A stronger mask does not pay at this horizon, and 0.1 stays the only rate tested with a
    positive read.

### Local screen: allocator size, standard attention and m/z encoding (2026-09-27)

The user's guidance was to start the set transformer small (16-32 inducing points, few heads) and
grow only on underfitting, and to try a standard transformer with m/z as the positional encoding.
Before these runs, every config had used at least 64 inducing points and 8 heads.

- **New key `masster_intensity_mz_positional_encoding`** (default off, uncommitted), in
  `CandidateIntensityTransformerHead._mz_encoding`.
  - It adds the transformer sinusoidal encoding to each candidate token, with the candidate's
    neutral mass in Da as the position (base 10000).
  - The shortest wavelength is 2π Da, so it resolves nominal mass but not the mass defect.
  - It works with both attention types. It is a non-persistent buffer, so the state dict and the
    initialization are unchanged.
  - 7 head tests and 2 model tests in `test_masster_attention_and_budget.py` and
    `test_masster_model.py` check the formula, the rotation property (a mass offset is a fixed
    rotation), the unchanged state and mass dependence.
  - Risk: the set of a molecule's candidate masses is itself a molecule fingerprint that
    attention can read.
- **Runs** (same setup as above):

| run | change from `c0` | params | tail 19-23 | best (epoch) | train loss 19-23 | best-ckpt train-eval |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ip16_h4` | 16 inducing points, 4 heads | 9.2M | 0.6338 | 0.6375 (22) | −3.351 | 0.8846 |
| `fa_h4` | `attention: full`, 4 heads | 8.1M | **0.6381** | 0.6419 (23) | −3.338 | 0.8877 |
| `fa_h4_mz` | `fa_h4` + m/z encoding | 8.1M | stopped at epoch 18 | | −3.338 (12-17) | |

- **`ip16_h4` does not underfit.** Train loss equals both controls' (−3.351), and val is +0.0043
  over the control mean. By the start-small rule it stays at 16. It does not narrow the gap: train
  fit is unchanged.
- **`fa_h4` is the best local run so far:** +0.0086 over the control mean, with 1.1M fewer
  parameters, 53.6 min against 57. It is still rising at epoch 23, with val loss 0.03 below the
  controls'. Its initialization differs from `c0`'s, so read it against the seed spread (0.0016
  on this tail). On merged, `a2_full_attention` was +0.0030 with 8 heads; this run uses 4 heads,
  on unmerged.
- **`fa_h4_mz` was stopped at epoch 18 of 24** when the user moved the work to Fir, so it has no
  tail and no train-eval pass. Another session relaunched it, and that run was stopped too; the
  18-epoch log is `logs/fa_h4_mz_sigterm_ep18.log`.
  - Over epochs 12-17 it is +0.0058 over `fa_h4` (5 of 6 above), and +0.0104 over epochs 0-5.
  - Its train loss is also lower, −3.338 against −3.261 over epochs 12-17, so the encoding raises
    train fit too. Whether it widens the gap is unmeasured.
- **Epoch-23 gap** (`eval_matched.py`, run by another session):
  - `ip16_h4`: val 0.6359, train-eval 0.8883, gap 0.2523.
  - `fa_h4`: val 0.6419, train-eval 0.8877, gap 0.2458.
  - The controls' range is 0.2436-0.2485, so neither narrows the gap. Their gains are val gains
    with train fit unchanged or up.
  - That run rewrote `eval_matched.json` with these two runs only. The earlier arms' numbers are
    in the tables above.

## Set-message substitution arm `sm1` (built 2026-09-27, not submitted)

`gm1`'s recipe, training-time substitution with the route kept, moved from the encoder's global
message to the allocator's set message. Not submitted.

- **Why.** The allocator-route probe replaces each candidate's set message with its spectrum's
  mean at evaluation
  ([read](../../findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md#set-message-probe-on-c0-r2-nl1-and-nl2-2026-09-27)).
  - It is the most train-specific route measured on both f0 and `c0`: train drop 0.078 and
    0.070 against val 0.023 and 0.017 (3.4x and 4.2x), with excess gap closed 0.050 on each.
  - On `r2`, trained without global attention, the same route carries almost twice the
    train-specific fit (train drop 0.132, excess 0.096), and val's use barely rises (0.027).
    The lookup moved there when its encoder route was cut. That is why `r2` left the gap at 0.18.
- **Option.** `masster_set_message_substitution: p` (default 0).
  - In each induced allocator layer, `candidate_attention` returns the message each candidate
    receives from its set, before the residual add.
  - In training, each real candidate independently, with probability `p`, receives instead the
    detached mean of the messages over its own spectrum's real candidates. Padded slots are left
    out of the mean and never drawn.
  - It keeps the set summary and loses what the candidate retrieved from the set for itself, the
    part the probe found most train-specific.
  - No parameters. At 0 or in evaluation it draws no random numbers, so `c0` (`vcpho5qm`) is the
    control.
  - Full candidate attention rejects it: every memorization arm uses induced attention.
- **Why not the alternatives.**
  - The batch mean (`setmsg_popmean`) would also remove the set summary that val uses (val drop
    0.033 on `c0`).
  - Hiding keys (`set_drop`, the Step 3 set-dropout idea) is weak (excess 0.013): retrieval
    survives on the keys that remain.
  - `h1`'s element-wise dropout inside the allocator's attention and feed-forward layers failed
    at −0.0067. It is not targeted.
- **Code.**
  - `_substitute_set_message` and the `set_message_substitution` argument of
    `_InducedCandidateAttentionLayer` in `bond_fragment_heads.py`, passed through
    `CandidateIntensityTransformerHead`.
  - `MassterModel(intensity_set_message_substitution=...)` in `masster_model.py`, wired in
    `masster_pl.py` and declared in `config/template.yml`.
- **Config.** `sm1_set_message_substitution.yml` is `c0_control.yml` plus the key at 0.3 (`gm1`'s
  rate): seed 397623, 64 epochs, no EMA, checkpoints uploaded.
- **Checks.**
  - Tests are in section 8 of `tests/test_masster_memorization_route.py`, shared with `sd1`.
    Each rules out a broken variant:
    - each real candidate keeps its message or takes its own set's mean, never a padded slot's,
      drawn independently per candidate across 24 seeds;
    - a replaced candidate passes no gradient back through the mean;
    - the state dict is unchanged, the arm is bit-identical at evaluation and differs in
      training;
    - nothing is drawn when off or in evaluation;
    - invalid probabilities and full attention are rejected;
    - the model reaches every allocator layer, is its control at evaluation, and `MassterPL`
      passes the key to both layers.
  - The allocator, probe, MASSter model, PL and config test files pass: 708 passed, 13 skipped
    (CUDA-only, run on CPU), with `sd1` in. Ruff is clean on the changed files.
  - Through `load_config`, `sm1` differs from `c0` by the key and W&B naming only. Built through
    `MassterPL` on CPU, both have 9,212,861 parameters, with two induced layers at 0.3 in `sm1`.
  - Two-epoch smokes of the option and its control on `debug_tiny_masster_cut_features.yml`
    exit 0. They ran on CPU (`accelerator=cpu`, `masster_candidate_backend=cpu`) because the GPU
    was in use.
    - The arm's train primary loss moves off the control's: −0.8657 against −0.8932 at epoch 1.
    - The two-epoch val numbers are noise and are not read.
  - The submitter's config checks pass for `sm1`. Locally it stops at the missing `pyteomics`
    wheel before generating anything.
- **Reading.**
  - The 54-63 tail against `c0`. A single seed needs about +0.006 (two standard errors), or a
    second seed. `val_candidate_wrecall` must stay within 0.005.
  - On the best checkpoint, the eval-mode train/val gap: train fit should fall while val rises,
    as for `k1`.
  - The allocator-route probe (`setmsg`, `set`, `cc`, `boundary`, `gsa`) re-run on the
    checkpoint. The set message's asymmetry should shrink without another route's growing; the
    boundary block and the pools are where the lookup would go next.
  - It is a long shot on the record: the route map's predictions have paid off once in three
    (`gm1`, unreplicated), and noise inside the allocator (`h1`) has hurt.
- **Generate** on Fir with `submit_masster_mh_ft_memorization_route.sh sm1`, or with `sd1` as
  stage `alloc`, after committing.

## Set dropout on the allocator's keys, arm `sd1` (built 2026-09-27, not submitted)

`sm1`'s companion on the same route. `sm1` masks the message a candidate receives from its set.
`sd1` masks which members the set summary is built from, so the summary, and what any candidate
can retrieve from it, changes from one exposure to the next. Not submitted.

- **Why a key mask, not dropping candidates.** Dropping candidates from the output would move a
  shared-m/z peak onto whichever isomer stays.
  - On average 2.75 candidate fragments share a formula.
  - At evaluation all isomers are present, so their summed probability would overshoot at every
    shared m/z value.
  - Hidden keys leave the output set and the loss exactly as in `c0`: each hidden candidate still
    queries the summary and is still predicted and scored.
- **Option.** `masster_set_key_dropout: p` (default 0).
  - In each induced allocator layer, `latent_attention` is where the 64 inducing tokens summarize
    the set from the candidates as keys.
  - In training, each real candidate independently, with probability `p`, is added to that
    step's key padding mask. It is drawn per candidate and layer.
  - A set whose real candidates would all be hidden keeps its first one visible, as in the
    probe's `drop_from_keys`. That is vectorized here, with no host sync.
  - No parameters. At 0 or in evaluation it draws no random numbers, so `c0` (`vcpho5qm`) is the
    control.
  - Full candidate attention rejects it.
- **Prior.**
  - Weaker than `sm1`'s. At evaluation, hiding half the keys (`set_drop`) closed only 0.012-0.013
    of excess gap on f0 and `c0`, against 0.050 for message substitution, because the inducing
    summary is robust to missing members. On `r2` it rose to 0.023.
  - Training with the mask is a different intervention from evaluating with it: the model can no
    longer count on a particular member being in the summary.
  - Its read is mainly relative: `sm1` against `sd1` tells whether the lookup sits in the message
    or in the membership.
- **Code.**
  - `_drop_set_keys` and the `set_key_dropout` argument of `_InducedCandidateAttentionLayer` in
    `bond_fragment_heads.py`, passed through `CandidateIntensityTransformerHead`.
  - `MassterModel(intensity_set_key_dropout=...)`, wired in `masster_pl.py` and declared in
    `config/template.yml`.
- **Config.** `sd1_set_key_dropout.yml` is `c0_control.yml` plus the key at 0.3: seed 397623, 64
  epochs, no EMA, checkpoints uploaded.
- **Checks.**
  - Section 8 of `tests/test_masster_memorization_route.py`:
    - the key mask keeps padding hidden;
    - it draws each real candidate on its own across 48 seeds;
    - it never empties a set: a one-candidate set is never hidden, and when a draw hides a whole
      set, the first candidate is the one kept visible;
    - a hidden candidate cannot move the inducing tokens, and a visible one does;
    - the checks shared with `sm1` (training-only, no parameters, nothing drawn when off,
      rejected settings, model and PL wiring) are parametrized over both options.
  - 17 cases in the section pass, and so do the related test files (708). The full suite passes
    on CPU with both options in: 4,450 passed, 107 skipped (CUDA-only), 2 xfailed.
  - Through `load_config`, `sd1` differs from `c0` by the key and W&B naming only. Built through
    `MassterPL` on CPU, both have 9,212,861 parameters, with two induced layers at 0.3 in `sd1`.
  - A two-epoch CPU smoke on `debug_tiny_masster_cut_features.yml` exits 0.
    - The train primary loss moves off the control's by less than `sm1`'s: −0.8854 and −0.8657
      against −0.8932 at epoch 1.
    - The two-epoch val numbers are not read.
  - The submitter's config checks pass for `sd1` and for the new stage `alloc` (`sm1` and `sd1`).
    Locally it stops at the missing `pyteomics` wheel before generating anything.
- **Reading.**
  - The 54-63 tail against `c0` (about +0.006 for a single seed, or a second seed), `wrecall`
    within 0.005, and the eval-mode gap on the best checkpoint.
  - Then against `sm1`:
    - `sm1` alone passing puts the lookup in the message;
    - both passing, and stacking, puts it in both;
    - neither passing says this route's training-time noise is not the lever, whatever the
      probe says.
- **Generate** on Fir with `submit_masster_mh_ft_memorization_route.sh alloc` (both), or `sd1`
  alone, after committing.

## Candidate-graph allocator arms `fa0`, `cg1` and `cg2` (proposed 2026-09-27, not built)

The user proposed building a graph over the 256 candidates and running a GNN on it in the
allocator (2026-09-27). Proposed only: no code, config or run exists, and nothing below is
measured on this lineage.

- **What exists.** The allocator already passes messages over a fixed candidate graph.
  - `c0`'s induced layers route every exchange through 64 inducing tokens. Full attention is the
    complete graph.
  - `_BiasedSelfAttention`, the full layer's attention, adds a pair bias to the attention logits
    ([`bond_fragment_heads.py:505-511`](../../../src/fragnnet/model/bond_fragment_heads.py#L505-L511)).
    A bias of −inf off the graph turns it into graph attention with `full`'s parameters.
  - The only pair bias that has run is the neutral-loss mass bias (`a3`, and inside `a7`). The
    induced layer rejects pair biases.
- **Why, on the record.**
  - The set message is the most train-specific route (`c0`: 4.2x, excess gap closed 0.050). On
    `r2` it carries almost twice the train-specific fit
    ([probe](../../findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md#set-message-probe-on-c0-r2-nl1-and-nl2-2026-09-27)).
  - A graph limits which candidates exchange messages. The hypothesis: comparative scoring needs
    related competitors, and a per-molecule lookup needs the global summary. The mask removes the
    second and keeps the first.
  - As a fix for comparison the record is weak:
    - On the merged lineage, full attention was +0.0030 (hold) and the mass bias on top −0.0002
      (null)
      ([arms §3](../../findings/masster/masster_architecture_arms_epoch128_2026_09_22.md#3-a2_full_attention-and-b1_bond_sum-read-at-the-same-screen)).
      Cut adjacency was +0.0032 (hold, §2).
    - Misplaced mass is 0.90-1.08x chance in every relation class against the support-preserving
      null
      ([displacement](../../findings/masster/masster_intensity_displacement_phase0_2026_09_24.md#the-misplacement-looks-local-only-against-the-wrong-null)).
      The one structured part is a containment relation: below NCE 60 the heavier member of a
      nested pair holds the mass.
    - The train ranking gap is 0.011.
  - Against the plan's premise: a mask holds at evaluation, so it is closer to a removal than to
    noise. No removal has helped, and `r2` rerouted
    ([audit](../../findings/masster/masster_memorization_attempts_audit_2026_09_27.md)). The prior
    is weak; the probe read is what the arm buys.
- **Edges.** Built per spectrum from the unique-fragment atom memberships
  ([`_unique_component_memberships`](../../../src/fragnnet/model/masster_model.py#L2334)). With
  `M` the candidates-by-atoms membership, `M Mᵀ` gives every pair's shared atoms in one product per
  bucket.
  - Self-loops.
  - Direct containment, both directions: `A ⊂ B` with no candidate strictly between. The
    containment matrix times itself finds the pairs with one between.
  - Complements: disjoint and together the whole precursor, the two sides of one acyclic cut.
  - Left out:
    - same-formula pairs: the mass bias's 0 Da centre already weights near-equal masses, and it
      was null;
    - sibling edges: every single-cut fragment is a child of the precursor, so they would join all
      of them.
- **Traps.**
  - **The precursor is a hub.** It is always a candidate (the forced empty mask,
    [`masster_cuda_selection.py:234`](../../../src/fragnnet/frag/masster_cuda_selection.py#L234))
    and contains every other one. As a sender it would hand every candidate the global summary in
    one hop, rebuilding the set channel. So it receives and does not send.
  - **Reach.** Two layers reach two hops. If most of a set lies within two hops of most
    candidates, the mask restricts nothing.
  - **Multi-parent.** A fragment between two acyclic cuts lies inside both single-cut sides, and
    single cuts are forced, so it has two direct parents. Attention aggregates over in-neighbours
    as it is.
  - **No empty rows.** Every query row, padded ones included, keeps its self-loop, so no row is
    all −inf.
  - **Identity is not the edge's contribution.** The token already carries the complement pool and
    the context. An edge adds the neighbour's contextualized token.

**Arms.** Each is `c0` plus its keys: seed 397623, 64 epochs, checkpoints uploaded.

| arm | keys against `c0` | question |
| --- | --- | --- |
| `fa0_full_attention` | `masster_intensity_transformer_attention: full` | The control: the graph arms need full attention, and no `[M+H]+`/FT config sets it. It is also the first probe read of an uncapped set channel. |
| `cg1_candidate_graph` | `fa0` + the graph mask (new key) | Does restricting the set channel to chemical neighbours shrink the lookup without rerouting? |
| `cg2_relation_bias` (optional) | `fa0` + one zero-initialized scalar per relation type, no mask (new key) | Do the relations help the comparison, with every route kept? |

- **Code (to write).**
  - Pass the unique memberships to the head beside `component_neutral_masses`, and densify them
    into each bucket's padded layout in `_contextualize_by_molecule`. The
    [attention-budget plan](masster_attention_budget_2026_09_21.md#not-implemented-here) left
    that step unbuilt for its containment bias.
  - The new keys default off and require `attention: full`.
  - `cg1` adds no parameters. `cg2` adds one scalar per relation type (direct parent, direct
    child, complement, other overlap, disjoint), shared across heads like the mass bias.
    Zero-initialized, it equals `fa0` at step 0.
  - Cost: on the merged lineage `a2` ran no slower than induced attention (0.817 against
    0.828-0.838 min/epoch), and the mass-bias arms cost 18% more. The graph build is unmeasured.
- **Reading.**
  - **Before building, no training:** the two-hop reach on `c0`'s candidate sets for the probe's
    60 train batches. If most candidates reach most of their set, rethink the edges first.
  - **Optional, offline:** the displacement transport with fragment-level containment. The
    finding's limits call its formula-level nesting coarse. A null there removes `cg2`'s premise,
    not `cg1`'s.
  - **`fa0` against `c0`:** full attention's own effect here (merged: +0.0030), and the probe on
    its checkpoint.
  - **`cg1` and `cg2` against `fa0`:**
    - the 54-63 tail (about +0.006 for one seed), with `wrecall` within 0.005;
    - the eval-mode gap on the best checkpoint;
    - then the probe. For `cg1` the set message's asymmetry must shrink with neither `gsa` nor the
      boundary block growing.
  - The probe's `setmsg` hook reads `candidate_attention`'s output, which the full layer has too
    ([hook](../../../scripts/debug/debug_pooled_embedding_memorization.py#L636-L667)). It is
    untested there.
  - Compare by wall-clock as well as by epoch.
  - **If `cg1` reroutes,** try the training-only form, the audit's working pattern: graph-only
    attention with probability p per layer in training, full attention at evaluation.

## Restricted allocators: mass window, no attention, smaller set transformer (built 2026-09-27, not run)

After the m/z-encoding arm, the user asked to try a less powerful allocator: a smaller set
transformer and mass-local attention. They also asked whether an MLP or a GNN would do. Code,
tests and a local queue are built. Nothing is submitted or run.

- **New options** (default off, uncommitted), in `CandidateIntensityTransformerHead`
  ([`bond_fragment_heads.py`](../../../src/fragnnet/model/bond_fragment_heads.py)):
  - `masster_intensity_mass_window: w` (Da; 0, the default, attends over the whole set). It
    needs `attention: full`.
    - Candidate i attends to j only when |m_i − m_j| ≤ w, on neutral masses, i itself included.
      It is a −inf pair bias on the attention logits (`_mass_window_bias`), so it adds no
      parameters and leaves the state dict unchanged. It stacks with the mass pair bias.
    - Padded query rows keep every key, so no row is all −inf. A test shows that without this,
      padded buckets go non-finite.
  - `masster_intensity_transformer_attention: none` is the MLP.
    - Each layer keeps only its feed-forward sublayer, with the same width, dropout and norm
      placement. Each candidate is scored from its own token, and candidates compete only
      through the output softmax.
    - Its tokens skip the per-molecule padding. Its state dict is `full`'s without the attention
      keys.
- **The GNN is `cg1`** ([above](#candidate-graph-allocator-arms-fa0-cg1-and-cg2-proposed-2026-09-27-not-built)).
  - Attention over a candidate graph is full attention with −inf off the graph. The mass window
    is that mask with edges |Δm| ≤ w, so `cg1` can reuse the path.
  - `cg1` still needs the atom memberships passed into each bucket.
- **Window reach, measured.** `output/local_encoder_noise_2026_09_27/mass_window_reach.py`, on
  `fa_h4`'s last checkpoint and 2,560 local val spectra. The median set is 256 candidates (p10
  140) over a median 274 Da. Share of a candidate's own set within reach, mean over candidates:

| window (Da) | 1 hop | 2 hops |
| ---: | ---: | ---: |
| 5 | 0.067 | 0.078 |
| 10 | 0.088 | 0.159 |
| 20 | 0.205 | 0.338 |
| 30 | 0.294 | 0.500 |
| 50 | 0.435 | 0.687 |
| 100 | 0.701 | 0.910 |

- **Window choice.** With two layers, 50 Da still reaches 69% of the set and 100 Da 91%, so those
  windows restrict little. 30 Da covers H2O, NH3, CO and C2H4 in one hop and halves two-hop
  reach. 10 Da excludes every common neutral loss.
- **Arms.** Same setup as the local screens above: `c0_control.yml` plus overrides, run by
  `run_queue_allocator_restriction.sh`. `eval_matched.py` knows all of them.

| run | change | read against |
| --- | --- | --- |
| `fa_h4_mz` | full, 4 heads, m/z encoding (rerun; the first stopped at epoch 18) | `fa_h4` |
| `mw30_h4_mz` | `fa_h4_mz` + 30 Da window | `fa_h4_mz` |
| `mw10_h4_mz` | `fa_h4_mz` + 10 Da window | `fa_h4_mz`, `mlp_mz` |
| `mlp_mz` | `attention: none` + m/z encoding, the window-0 end | `fa_h4_mz` |
| `ip1_h4` | 1 inducing point, 4 heads | `ip16_h4`, `c0` |
| `ip4_h4` | 4 inducing points, 4 heads | `ip16_h4` |
| `ip16_h4_l1` | `ip16_h4` with 1 layer | `ip16_h4` |

- **`ip1_h4`.** With one latent, each candidate's query sees a single key, so every candidate in a
  spectrum receives the same set message. It is the learned, undetached form of `sm1`'s
  substitution at rate 1.
- **Priors.**
  - `ip16_h4` left train fit and the gap unchanged, so a smaller set transformer is not expected to
    narrow the gap. Of the size arms, only `ip1_h4` removes per-candidate retrieval.
  - The [plan review](../../findings/masster/masster_memorization_plan_review_2026_09_27.md#a-smaller-next-step-sequence)
    warns that global information is already in the input tokens, so restricting edges cannot
    guarantee removing identity. No removal has helped so far (audit). Read the window arms as
    performance arms first, mechanism second.
  - `mlp_mz` prices candidate exchange on this recipe, the review's step 4 comparison. It is
    end to end, not on fixed candidates, so each arm's proposal can differ: read `wrecall`.
- **Reading.** As for the local screens: the 19-23 tail against its control (one-pair floor
  0.0016 on this tail), the epoch-23 gap from `eval_matched.py`, `wrecall`, and wall-clock.
- **Checks (2026-09-27).**
  - Tests in `test_masster_attention_and_budget.py`, `test_masster_model.py` and
    `test_masster_pl.py`:
    - a candidate outside the window cannot move another's output, and one inside can;
    - the exact mask values, padded rows included;
    - a window wider than the set equals full attention, with the same state dict;
    - padded buckets stay finite through backward;
    - under `none`, each candidate's output depends only on its own input, for both norm
      placements;
    - model and PL wiring.
  - Those files and `test_bond_fragment_heads.py` and `test_masster_memorization_route.py` pass:
    475 passed, 7 skipped (CPU).
  - **Real-data smoke:** 12 training steps each of `fa_h4_mz`, `mw30_h4_mz`, `mlp_mz` and
    `ip1_h4` on the local split, all exit 0.
    - It used `masster_candidate_backend: cpu`. The CUDA selection kernel failed to compile in the
      middle of another session's edit (SMARTS extra forced masks), so the queue must wait for
      that edit to land.
    - Parameters: 8,142,269 for `fa_h4_mz` and `mw30_h4_mz`, 7,614,909 for `mlp_mz`, 9,196,733
      for `ip1_h4`.
  - **Head-only cost**, forward plus backward on the GPU at 128 spectra × 256 candidates:

| head | params | ms/step |
| --- | ---: | ---: |
| `c0` (induced 64, 8 heads) | 2,423,564 | 86.1 |
| `ip16_h4` | 2,411,276 | 68.4 |
| `ip1_h4` | 2,407,436 | 65.5 |
| `ip16_h4_l1` | 1,357,068 | 40.8 |
| `fa_h4_mz` | 1,352,972 | 76.7 |
| `mw30_h4_mz` | 1,352,972 | 78.3 |
| `mlp_mz` | 825,612 | 30.5 |

- **Cost.** The window saves no compute: the mask still materializes every pair. The MLP head is
  2.8x faster than `c0`'s. A local `c0` step takes about 0.48 s, so the head is under a fifth of
  it.

## Batch 4 read from W&B (2026-09-28)

Six runs that finished with no read recorded. Tail = mean `val_cos_sim_0.01` over epochs 54-63.
The loss columns are same-window deltas of train-mode and eval-mode primary loss; coverage is the
same delta of `val_candidate_wrecall`. The seed spread is 0.0042.

| arm | run | control | tail | delta | above control | coverage | train loss | val loss |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| `s1` cosine LR | `nq2al58c` | `c0` (0.7276) | 0.7366 | **+0.0090** | 56 of 64 | +0.0013 | −0.043 | −0.035 |
| `s1` cosine LR | `nq2al58c` | `u2` (0.7392) | 0.7366 | −0.0026 | 0 of 64 | — | −0.041 | +0.009 |
| `s1_k1_ema` | `f2oacs52` | `k1_ema` (0.7421) | 0.7410 | −0.0011 | 2 of 64 | +0.0008 | −0.053 | +0.003 |
| `m2` sink dropout | `41mnwm9f` | `c0` | 0.7321 | +0.0045 | 60 of 64 | +0.0001 | −0.001 | −0.021 |
| `sm1` set-message substitution | `7kz04hra` | `c0` | 0.7292 | +0.0016 | 25 of 64 | +0.0001 | +0.038 | −0.004 |
| `sd1` set key dropout | `uxie2eko` | `c0` | 0.7319 | +0.0043 | 55 of 64 | +0.0000 | +0.011 | −0.026 |
| `am1` encoder atom mask | `dhn91qeg` | `c0` | 0.7286 | +0.0010 | 37 of 64 | −0.0001 | +0.032 | −0.014 |

- **The LR schedule passes against `c0` but is the same gain as EMA.** `s1` is +0.0090 over
  `c0`, 2.1x the seed spread, and fits train better (−0.043). It stays below `u2` at every epoch,
  and on top of `k1_ema` it adds nothing (−0.0011). Decay and EMA both remove late-training
  weight noise, so use one. EMA is the larger and needs no horizon, so the schedule is dropped
  from the stack.
- **`m2` and `sd1` are single-seed, inside the spread** (+0.0045, +0.0043). `m2` recovers m1's
  loss entirely (+0.0248 over `m1`). Neither is worth a stack slot before a second seed.
- **`sm1` and `am1` are null** (+0.0016, +0.0010). `am1`'s Fir tail does not reproduce the
  +0.0067 of its local screen.
- **`sm1` has no variant worth running (re-read 2026-09-28).** Its code matches the plan, so the
  null is not a bug. Its curve differs from every other arm on this page:
  - It is the only arm that starts behind `c0`: −0.0076 over epochs 0-9, −0.0053 over 10-19,
    and below `c0` at 37 of epochs 0-41. Val loss is also worse early (+0.084 at epoch 3), so
    what each candidate retrieves from its set carries fit that transfers to val, and
    substitution removes it. `sd1`, `gm1` and `m2` lead from the first epochs, and `k1` from
    epoch 10.
  - It pulls ahead only after `c0`'s val peak (epoch 42), and ends at `c0`'s best (0.7292).
    Against `c0`'s best-so-far over epochs 54-63 it is +0.0000, and best checkpoint against best
    checkpoint +0.0009. All of the tail gain is `c0`'s post-peak decline (0.0016), which EMA and
    best-checkpoint selection already remove.
  - So neither a lower rate nor a delayed start has room: each is capped by that same 0.0016.
    Stacking on `k1_ema` has less, since its post-peak decline is 0.0006.
  - On this route, `sd1` is the arm that could still pay off, with a second seed or a slot in
    the `i3` stack. The allocator-route probe on `sm1`'s checkpoint is still unrun. It can only
    show where the lookup went, and cannot change this verdict.
- Coverage moves by at most 0.0013, so none of this is candidate-set coverage.

## Stacking on the instrument input: `i3` and `i4` (built 2026-09-28, not submitted)

`i1_inst_ob_obf` (`i50ig7rf`, `n4` on the `OB` / `OB_F` labels) led `n4` by +0.021 mid-run
([instrument finding §8](../../findings/nist_orbitrap_instrument_ce_scales_2026_09_28.md#8-training-with-the-instrument-input-i1-and-i2-running-read-2026-09-28)).
These arms stack on it the levers that replicated on the `FT` labels, then the one architecture
key that passed the merged screen.

| arm | config | change | control | evidence on `FT` labels |
| --- | --- | --- | --- | --- |
| `i3` | `i3_k1_ema_ob_obf.yml` | `i1` + `k1`'s two 0.2 dropouts + `ema_decay: 0.999` | `i1` | `k1_ema` +0.0145 over `c0`; EMA and `k1` each replicate at seed 811237 |
| `i4` | `i4_k1_ema_prenorm_ob_obf.yml` | `i3` + `masster_intensity_norm_placement: pre` | `i3` | `a1_prenorm` +0.0056 on the merged 128-epoch screen; never run unmerged |

- **Left out.** The cosine schedule adds nothing on top of EMA (Batch 4 read). `gm1`, `nl1`,
  `m2` and `sd1` are single-seed passes inside the spread.
- **Expected size.** Part of `i1`'s gain may be the same generalization gap `k1` and EMA act on,
  so `i3` may add less than +0.0145. Read it at +0.005 (the seed spread) on one seed.
- **Checks.**
  - Through `load_config`, `i3` differs from `i1` by the three keys, and `i4` from `i3` by
    `masster_intensity_norm_placement`, plus W&B naming. `i3` equals `k1_ema` apart from
    `int_embedder: fourier` (and the relabelled data).
  - A local 2-epoch smoke of `i4` on a 512-spectrum subsample of the relabelled NIST20 (batch 64)
    exits 0, val cosine 0.094 → 0.233. Its checkpoint holds the EMA state, a 2 × 32 instrument
    embedding, `pre` placement and both dropouts at 0.2.
  - `bash -n` passes on the submitter.
- **Generate** on Fir with `submit_masster_mh_ft_memorization_route.sh stack` (`i3` and `i4`).
  The cluster needs the relabelled `data/proc/nist20/spec_df.pkl` (md5
  `bdc009bbeaf6a38d126326d127630cad`), as for `i1`.

### Memorization read on `i1` (2026-09-28)

This is the Step 0 probe on `i1`'s checkpoints (epochs 59 and 63), paired with `c0`'s on the same
spectra ([finding](../../findings/masster/masster_i1_memorization_read_2026_09_28.md)).

- **The gap narrows by 0.025 on the probe and 0.028 on the full val split, at matched epoch 63.**
  Train fit rises by +0.003. No arm here had done both.
- **Why.** Only 7 of 12,668 train molecules have both instruments, so on the `FT` labels the
  identity routes stood in for the instrument.
  - `c0`'s Lumos train spectra leaned harder on the set message and global attention.
  - Its Lumos val spectra were right only at Lumos-typical NCE values.
  - Set-message damage ratio: 4.20 → 2.44. Global attention: 2.53 → 2.23. The boundary block is
    unchanged.
- **What it changes here.**
  - Every earlier arm was read on the `FT` labels, where part of the "lookup" is this instrument
    proxy, and on a probe val sample that is 90% Elite.
  - About 86% of the gap remains on `i1`, with the lookup signature intact. `i3` puts `k1` and EMA
    on that remainder.
- **Gates.** 1, 3, 4 and 5 pass (3 against `c0`; `n4` was not run, though a patched checkpoint
  can run it). Gate 2 was not run.

## Full attention and the m/z encoding on `i1`: `fa1` and `mz1` (built 2026-09-29, not submitted)

The user asked whether a sequence transformer had been tried on the intensity head (2026-09-29).
The allocator is already a candidate transformer, but a set transformer: `i1` routes every
exchange through 64 inducing tokens, and its tokens carry no position. The sequence form,
standard attention with each candidate's neutral mass as its position, is the local `fa_h4_mz`,
which stopped at epoch 18
([local screen](#local-screen-allocator-size-standard-attention-and-mz-encoding-2026-09-27)).
Full attention has run only on the merged lineage. This pair runs both on Fir, on `i1`.

| arm | config | change | control | evidence so far |
| --- | --- | --- | --- | --- |
| `fa1` | `fa1_full_attention_ob_obf.yml` | `i1` + `masster_intensity_transformer_attention: full` | `i1` (`i50ig7rf`) | merged `a2` +0.0030 (hold); local `fa_h4` (4 heads) +0.0086 |
| `mz1` | `mz1_full_attention_mz_ob_obf.yml` | `fa1` + `masster_intensity_mz_positional_encoding: true` | `fa1`, same stage | local `fa_h4_mz` +0.0058 over `fa_h4` at epochs 12-17 (5 of 6), no tail |

- **Heads.** Both keep `i1`'s 8 heads, so each arm is one key from its control. The local screen
  used 4. Fewer heads leave the parameter count unchanged.
- **`fa1` is the proposed `fa0` moved to `i1`.** It is also arm F of the
  [hierarchical fragment graph plan](masster_hierarchical_fragment_graph_plan.md#experiments-that-identify-each-contribution),
  whose masked arms must keep its head count, so 8.
- **Parameters and initialization** (models built at seed 397623, no data):
  - `i1` has 9,215,701 parameters, 2,457,355 in the intensity head. `fa1` and `mz1` have
    8,145,109 and 1,386,763.
  - Outside the intensity head, `fa1` initializes bit-identically to `i1`. Inside it, 14 shared
    tensors differ and 25 of `i1`'s keys (the latent block and inducing tokens) are gone. So
    read `fa1` against the 0.0042 seed spread.
  - `mz1`'s state is bit-identical to `fa1`'s, so that pair is init-matched.
- **Memory on the 3g.40gb slice**, from W&B system metrics (NVML, allocator cache included):
  - Merged `a2` (full attention, 8 heads, batch 512) had a median of 28.0 GiB, against 30.4 GiB
    for its induced control `c4a_base_s1` (`5qws8g9x`), and finished 128 epochs.
  - `i1` runs closer to the ceiling: median 19.7, p90 36.9, max 39.5 GiB.
  - So the profile stays. If `fa1` runs out of memory, move it to a larger profile at the same
    batch rather than change the batch.
- **Cost.** `a2` ran at its control's speed (0.817 against 0.828-0.838 min/epoch). Head-only,
  `fa_h4_mz` took 76.7 ms per step against `c0`'s 86.1 (128 spectra × 256 candidates). `i1` took
  3.32 h for 64 epochs, and the stage keeps the 5 h limit.
- **Risk for `mz1`.** A molecule's set of candidate masses is itself a fingerprint attention can
  read, so the encoding may widen the train/val gap as it raises train fit. `fa_h4_mz`'s lower
  train loss (−3.338 against −3.261) cannot tell the two apart.
- **Reading.** For each arm, the 54-63 tail of `val_cos_sim_0.01_epoch/mean` against its control,
  at least +0.005 to count on one seed; `wrecall` within 0.005; minutes per epoch. Read `mz1`
  against `i1` too. If `mz1` passes, compare the eval-mode gap on its best checkpoint with
  `fa1`'s, for the fingerprint risk.
- **Checks (2026-09-29).**
  - Through `load_config`, `fa1` differs from `i1` by the attention key, and `mz1` from `fa1` by
    the encoding key, plus W&B naming.
  - On CPU: `test_every_tracked_config_still_resolves`, `test_masster_attention_and_budget.py` and
    `test_bond_fragment_heads.py` pass (97 passed, 2 CUDA-only skipped), and so do the model and
    PL tests that train through full attention and the encoding (10 cases).
  - The submitter's `attn` stage passes `bash -n` and its per-arm config checks. Locally it stops
    at the wheel check, because `wheels/` is only in the cluster checkout.
  - **Not run: a real-data smoke.** The first attempt got SIGKILL in setup, while another
    session's local `i1` controls held the GPU and RAM and swap were nearly full; it was not
    retried. The CPU fallback on `debug_tinyv4` cannot run, because that dataset still carries
    `FT` labels, which the loader refuses. Moot since both arms ran 64 epochs on Fir
    ([Batch 5 read](#batch-5-read-from-wb-i3-i4-fa1-and-mz1-2026-09-29)).
- **Unchanged:** the local restricted-allocator queue is still unrun. Its `fa_h4_mz` rerun
  remains the control for its window arms.
- **Generate** on Fir with `submit_masster_mh_ft_memorization_route.sh attn` (`fa1` and `mz1`).
  The cluster needs the relabelled `data/proc/nist20/spec_df.pkl` (md5
  `bdc009bbeaf6a38d126326d127630cad`), as for `i1`.

## Dual CE on `i1`: `i5` (built 2026-09-29, not submitted)

`i5_dual_ce_ob_obf.yml` is `i1` with the recorded eV as a second CE input. It sets
`spec_params.ace: true` and the new `spec_params.fill_missing_ace: false`. Its control is `i1`
(`i50ig7rf`, tail 0.7607).

- **Why a loader switch.** By default the loader fills a missing eV from NCE as
  NCE × m/z / 500, the Elite convention.
  - On this split every `OB` train spectrum records eV, a median 0.45 eV below the conversion.
  - 67% of `OB_F` train spectra (28,766 of 42,712) record none.
  - The `OB_F` spectra that do record eV sit a median 12.6 eV above the conversion.
  - Filled, two-thirds of Lumos spectra would get an eV about 12 eV below recorded Lumos eV at the
    same energy, and the model could not tell them apart. The
    [instrument finding](../../findings/nist_orbitrap_instrument_ce_scales_2026_09_28.md#implications)
    asks for this to be fixed before any dual-CE arm.
- **The switch.** `spec_params.fill_missing_ace` is declared in `config/template.yml`, default
  `True`, so every other config loads as before.
  - `False` leaves a missing eV at −1, which the `embed` / `fourier` table reads as a learned
    extra token.
  - NCE is still filled from eV either way.
  - `ace_max` stays at `i1`'s 300: no recorded `[M+H]+` train eV exceeds 275.
- **Checks.**
  - Through `load_config`, `i5` differs from `i1` by the two `spec_params` keys and W&B naming.
  - On the real train split with the switch off, exactly the 28,766 `OB_F` spectra without eV
    read −1, no `OB` spectrum does, and every recorded eV is unchanged.
  - Four new tests in `tests/test_split_filters.py`; the suite passes on CPU.
  - Not run: a GPU smoke. Another session's local `i1` training held the GPU. The same dual-CE
    context trained in a 2-epoch smoke of `n4` + eV (0.076 → 0.232).
- **Read.**
  - The tail against `i1`, at +0.005 on one seed; coverage within 0.005.
  - Then `OB_F` val cosine and precursor-fraction bias, split by recorded against missing eV
    (`collect_masster_preds.py`, then `ladder_bias.py` and `nist20_raw_labels.py` with tag `i5`).
- **Generate** on Fir with `submit_masster_mh_ft_memorization_route.sh i5`.
  - The cluster needs the relabelled `spec_df.pkl`, as for `i1`.
  - It also needs a checkout with the loader change.

## Batch 5 read from W&B: `i3`, `i4`, `fa1` and `mz1` (2026-09-29)

All four finished 64 epochs on Fir; the `i3`/`i4` section above predates their submission. Same
method as Batch 4: the full row stream, forward-filled per epoch; tail = mean
`val_cos_sim_0.01_epoch/mean` over epochs 54-63. The method reproduces the recorded tails of `c0`
(0.7276), `n4` (0.7329) and `i1` (0.7607). The loss columns are tail deltas of
`train_primary_loss_epoch/mean` and `val_primary_loss_epoch/mean`, where lower is better. The
seed spread is 0.0042.

| arm | run | control | tail | delta | above control | coverage | train loss | val loss | min/epoch |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `i3` `k1` + EMA | `xypkkfti` | `i1` (0.7607, 3.07 min/epoch) | 0.7710 | **+0.0102** | 64 of 64 | +0.0003 | +0.071 | −0.053 | 3.10 |
| `i4` + pre-norm | `rd9rsdqx` | `i3` | 0.7729 | +0.0019 | 62 of 64 | +0.0004 | +0.001 | −0.019 | 3.05 |
| `fa1` full attention | `0vn8m2gr` | `i1` | 0.7651 | +0.0044 | 58 of 64 | −0.0001 | +0.012 | −0.026 | 2.99 |
| `mz1` + m/z encoding | `tck857p7` | `fa1` | 0.7634 | −0.0017 | 14 of 64 | −0.0003 | −0.014 | +0.005 | 2.99 |

- **`i3` passes: +0.0102, 2.4x the seed spread, above `i1` at every epoch.** `k1` and EMA carry
  to the instrument labels (`k1_ema` was +0.0145 over `c0` on `FT`). Train fit drops as val loss
  improves. It reaches `i1`'s best (0.7622) at epoch 20, 1.1 h, against `i1`'s epoch 59, 3.1 h.
- **`i4` is +0.0019 over `i3`, a single-seed hold.** Pre-norm's merged +0.0056 does not reproduce
  here. It is the best tail on the lineage: 0.7729, +0.0122 over `i1`.
- **`fa1` is +0.0044 over `i1`: consistent, but just under the one-seed +0.005 rule.**
  - It leads in every ten-epoch window (+0.0026 to +0.0066) and in 58 of 64 epochs. Its best,
    0.7674, is its last epoch.
  - It fits train less and val better, so it narrows the gap rather than adding fit. On the merged
    lineage the same key was +0.0030.
  - It is cheaper: 2.99 against 3.07 min/epoch, 12% fewer parameters, and a GPU peak of 22.3 GiB
    against `i1`'s 39.5 (NVML). It reaches `i1`'s best in 1.96 h against 3.11 h.
  - It is 0.0059 below `i3`. Whether it adds on top of the dropout and EMA is the open question.
  - Hypothesis, not tested: on `c0`, the induced layer's set message, read back from the inducing
    tokens, was the most train-specific route (4.2x). Full attention has no shared latent summary.
- **`mz1` rejects the m/z encoding: −0.0017 against `fa1`, below it in 50 of 64 epochs.**
  - It leads early, as the local screen did (+0.0067 over epochs 0-5). It is level over epochs
    10-19 and behind in every later window (−0.0041, −0.0026, −0.0015, −0.0017).
  - Train loss is better and val loss worse: the fingerprint risk above. The encoding buys train
    fit that does not transfer.
  - So the local `fa_h4_mz` lead at epochs 12-17 was the early phase, and that run stopped before
    the crossover.
  - Against `i1` it is +0.0026, all of it from full attention.
- Coverage moves by at most 0.0004, so none of this is candidate-set coverage.
- **Also finished, unread until now: `i2_qet_ob_obf` (`jshvk4gv`)**, `n1`'s `qet` energy input on
  the `OB` / `OB_F` labels. Its tail is 0.7592, −0.0015 against `i1`, inside the seed spread; it
  is above `i1` in 10 of 64 epochs, with train and val loss unchanged (±0.0002). The mid-run −0.004
  (epochs 10-14) narrowed to that: the smooth energy input costs little once the instrument is an
  input, and gains nothing.
- **Consequences.**
  - Drop the m/z encoding on its own. The user's follow-up pairs it with the neutral loss
    ([`mznl1`/`nlpe1`](#neutral-loss-as-a-second-position-mznl1-and-nlpe1-built-2026-09-29-not-submitted)).
    The local queue's `mw30_h4_mz`, `mw10_h4_mz` and `mlp_mz` all carry it. The mass window does
    not need it, so if they run, they run without it.
  - `fa1` is arm F of the hierarchical fragment graph plan, now with F − B = +0.0044 on one seed.
  - Next architecture arm, not built: `i4` + full attention, against `i4`. It would be the first
    run of pre-norm with full attention alone; the merged `a7` also carried CE-FiLM and both pair
    biases.

## Neutral loss as a second position: `mznl1` and `nlpe1` (built 2026-09-29, not submitted)

The user's proposal after the Batch 5 read: give full attention the neutral loss as a position,
beside the m/z. `mz1` lost to `fa1` (−0.0017), with train loss better and val loss worse. A
candidate's mass is specific to its molecule, but its loss from the precursor recurs across
molecules.

- **New key `masster_intensity_nl_positional_encoding`** (default off, uncommitted), in
  `CandidateIntensityTransformerHead`:
  - Each candidate token gets `mz1`'s sinusoidal encoding with its neutral loss M − m in Da as the
    position. M is the precursor's intact neutral mass (`precursor_neutral_masses` of the
    candidate inputs), so the precursor's loss is 0 and a water loss's 18.011.
  - With the m/z encoding also on, the two split the token width: m in the first half, M − m in the
    second, each a full sinusoidal ladder over its half. Summed over the same dimensions they would
    not separate: per frequency w, PE(m) + PE(M − m) = 2 cos(w (m − M/2)) (sin(w M/2), cos(w M/2)),
    whose direction depends on M alone.
  - Alone, the loss takes the full width. The m/z encoding alone is unchanged, so `mz1`'s
    checkpoint behaves as trained.
  - No parameters; the state dict and initialization are unchanged. Both head call sites
    (unique-fragment and pathway-token mode) pass the precursor masses.

| arm | config | change | control |
| --- | --- | --- | --- |
| `mznl1` | `mznl1_full_attention_mz_nl_ob_obf.yml` | `fa1` + both encodings, 128 dimensions each | `fa1` (`0vn8m2gr`); also `mz1` (`tck857p7`) |
| `nlpe1` | `nlpe1_full_attention_nl_ob_obf.yml` | `fa1` + the loss encoding, all 256 dimensions | `fa1` |

- **Why `nlpe1` as well.** It separates the two halves of the user's `mznl1`: `mz1` already prices
  the m/z alone. The m/z code in `mznl1` is half as wide as `mz1`'s, so `mznl1` − `mz1` is not a
  pure "add the loss" contrast.
- **Init-matched.** `fa1`, `mz1`, `mznl1` and `nlpe1` all have 8,145,109 parameters and
  bit-identical initial states at seed 397623.
- **Risk.** A molecule's set of losses is as much a fingerprint as its set of masses. What differs
  is that one token's position is shared across molecules. Read the train/val loss split, as for
  `mz1`.
- **Reading.** The 54-63 tail against `fa1`, +0.005 on one seed; `wrecall` within 0.005 of `fa1`;
  the train and val loss deltas. Read `mznl1` against `mz1` too.
- **Checks (2026-09-29).**
  - Through `load_config`, `mznl1` differs from `fa1` by the two encoding keys and `nlpe1` by one,
    plus W&B naming.
  - 13 new test cases: the formula on the loss; an exact token offset of PE(M − m) from each
    spectrum's own precursor, under induced and full attention; the half-and-half layout; an
    unchanged state dict for every attention type; outputs of one spectrum untouched by another's
    precursor mass; input validation; and, in the model, the precursor masses reaching the head
    in all three modes, with the precursor present as a loss-0 candidate.
  - **Checkpoints from before the key** load with it off. `get_pl_hparams` and the Lightning
    loader's legacy marker backfill `False`, as for the attention-topology key, with one more test
    case.
  - On CPU, the head, model, PL, config and hparam-migration test files pass: 637 passed, 13
    CUDA-only skipped. Ruff passes.
  - **Real-data GPU smoke** (`output/smoke_attn_2026_09_29/run_smoke.sh`, 2 epochs on 512
    relabelled NIST20 spectra, batch 64): both exit 0. Val cosine rises 0.152 → 0.333 (`mznl1`)
    and 0.144 → 0.332 (`nlpe1`). The checkpoints record both keys and hold no frequency buffers
    (155 tensors, as `fa1`).
  - The submitter's `pos` stage passes `bash -n` and its per-arm config checks.
  - **Side finding, not fixed:** the hparams-driven loaders (`MassterPL.load_from_checkpoint`,
    `get_pl_hparams` plus the constructor) already fail on `i1`'s Fir checkpoint, whose hparams
    lack `masster_intensity_mass_window`. They also fail on this week's local checkpoints, which
    lack the uncommitted `masster_fragment_connectivity_*` keys. The probes are unaffected: they
    build the model from the template and config, then load the weights strictly.
- **Generate** on Fir with `submit_masster_mh_ft_memorization_route.sh pos` (`mznl1` and `nlpe1`).
  The cluster needs the relabelled `spec_df.pkl` and a checkout with the new key.

## Mass-window ladder on `fa1`: `mw30`, `mw10` and `mlp1` (built 2026-09-29, not submitted)

The user asked for the three restricted allocators of the
[2026-09-27 local queue](#restricted-allocators-mass-window-no-attention-smaller-set-transformer-built-2026-09-27-not-run)
as Fir runs. That queue defined them on `c0` with 4 heads and the m/z encoding. They are rebuilt
on the current lineage without the m/z encoding, which lost on Fir (`mz1` −0.0017 against `fa1`).

| arm | config | change from `fa1` | parameters |
| --- | --- | --- | ---: |
| `mw30` | `mw30_mass_window30_ob_obf.yml` | `masster_intensity_mass_window: 30.0` | 8,145,109 |
| `mw10` | `mw10_mass_window10_ob_obf.yml` | `masster_intensity_mass_window: 10.0` | 8,145,109 |
| `mlp1` | `mlp1_no_attention_ob_obf.yml` | `masster_intensity_transformer_attention: none` | 7,617,749 |

- **What they change.**
  - The window restricts who each candidate attends to (a −inf pair bias; no parameters). The
    final softmax over the spectrum is unchanged, so competition for intensity stays global.
  - `mlp1` removes candidate exchange entirely.
  - The ladder is the "restricted communication without chemistry" control for a
    chemistry-defined grouping (containment graph, CFM-like parent/child competition). Mass
    windows do not match the failing decisions:
    - sibling pairs differ by far more than 30 Da;
    - many nested losses are 50-250 Da
      ([base-peak table](../../findings/masster/masster_i1_base_peak_errors_2026_09_28.md)).
- **Control.** `fa1` (`0vn8m2gr`, tail 0.7651), read against `i1` (`i50ig7rf`) too.
- **Read.**
  - The tail over epochs 54-63 against `fa1`, with a +0.005 one-seed rule.
  - `wrecall` within 0.005.
  - Offline, the base-peak probes' train/val deficit on nested and overlapping pairs at novelty
    < 0.6.
- **Submit.** Stage `window`: `submit_masster_mh_ft_memorization_route.sh window`. The key is in
  HEAD; the cluster needs the relabelled `spec_df.pkl`.
- **Checks.** All three configs build, and each differs from `fa1` only in W&B naming and its key.
  Locally, the submitter's config validation passes. It stops at the wheel check because
  `wheels/` is absent on this workstation.
