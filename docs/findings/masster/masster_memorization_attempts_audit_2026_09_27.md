# Audit of every attempt to close MASSter's memorization gap

> **Status:** FINDING · **Updated:** 2026-09-27
> Read-only audit of every training arm, probe and post-hoc lever aimed at MASSter's train/val
> gap, 2026-09-16 to 2026-09-27, on both lineages. Each claim was re-derived from code (current
> tree and git history), the configs as run (W&B `run.config`) and W&B histories, not from the
> docs. No training, no GPU. W&B read 2026-09-27, 06:00-08:00 UTC; four arms were still running.
> **The corrections below are applied in the docs they name** (2026-09-27).
>
> - **Nothing tested closes the gap.** The eval-mode train/val cosine gap is about 0.18 in every
>   arm read. The best arm (`k1`) narrows it by about 0.015, and `m1` narrows it only by
>   underfitting.
> - **Three levers are real:** weight EMA (both seeds, both lineages), `k1` key dropout (both
>   seeds) and LR decay (merged only; never run on unmerged). EMA and decay lift train and val
>   together, so they reduce variance, not memorization.
> - **Removing a route never helped:** `r0`, `r2`, `nl2`, `g1`, `c4a_no_bbe` and the capacity
>   cuts. Noise on a route that other routes back up helped a little: `k1`, and `gm1`
>   unreplicated. Noise on a model already stripped of routes underfits (`nl3`).
> - **The record needs corrections.** `nl1` finished at +0.0037, not null. `nl2` underfits
>   against its same-code control. `sp0`/`sp1` were resumed from pre-fix weights, not
>   resubmitted. `m2` tracks `c0`, not `m1`. `n2`/`n3` fail like `n1`. Three merged arms marked
>   "never run" ran. [List](#corrections-to-the-record).
> - **The +0.003 gate is about one standard error.** A null arm clears it 9-24% of the time.
>   Among the fourteen single-change arms against `c0`, chance predicts one to three passes;
>   three passed, and only `k1` is replicated. [Noise floor](#noise-floor-and-the-gate).
> - Gates 2-4 (ranking decomposition, route map on the arm's checkpoint, novelty bins) have not
>   been run on any arm.

## Scope and method

Nine read-only auditors each took one slice and re-derived it from primary sources:

- the Step 1 route arms;
- the global-message arms;
- averaging, schedule, decay and replicates;
- the light-side token and the sinks;
- the NCE and shared-proposal arms, with the sampler bug;
- provenance and code drift;
- the diagnostic instruments;
- the pre-plan attempts;
- the untried levers.

The claims below that change a recorded conclusion were then re-pulled and re-read directly:
the final tails of `nl1`, `nl2`, `n1`-`n3` and `m2`, the `sp0`/`sp1` histories, steps per epoch
and resume path, the `r0` construction order and the probe's val batch order.

- **Metric.** Tail = mean of `val_cos_sim_0.01_epoch/mean` over epochs 54-63 (unmerged, 64 epochs)
  or 118-127 (merged screen, 128 epochs). Running arms are compared over their latest ten epochs
  shared with the control.
- **Controls.** Unmerged `[M+H]+`/FT: `c0` (`vcpho5qm`, tail 0.7276) for the memorization-route
  group `nist20_mh_ft_memorization_route_2026_09_24`, and `f0` (`3bej6fq7`, 0.7298) for the
  ranking-transfer arms. Merged MA/MI: the `c4a_base_s1`/`s2` mean (0.7270, gate ±0.005), or the
  regularization sweep's control mean at epoch 64 (0.7160).
- **What reproduces.** Every tail, delta, epochs-above-control count and coverage delta in the
  memorization-route and ranking-transfer tables matches W&B to 4 decimals, as do the merged
  numbers the plan cites. The 29 arm-vs-control config diffs, as run, show only each arm's
  documented keys. The MASSter rename and the QET commits change no default for the other arms,
  and the sampler fix reaches only `sp0`/`sp1`. About 25 route-map, side-of-cut, set-message and
  context-probe numbers recomputed from the stored CSVs match their findings exactly.
- **Not checked.** Gates 2-4 need GPU passes. The eval-mode gap table's raw outputs were never
  stored. `nl3`'s traceback was never uploaded. The Fir working trees at submission time are not
  visible (config fingerprints across all 33 runs show no anomaly).
- **Auditor claims that failed verification, and are left out.**
  - "The energy probe's shift-landing table cannot be reproduced": it can. Recomputed from the
    saved per-spectrum probe pickle and `spec_df`, it matches to the digit.
  - "The 4 h campaign plan says `c4d_drop02` never produced data": that column gives the reason
    the arm was rerun, not a claim about the rerun.
  - "`sp0`/`sp1` are still −0.21 below `c0`": that number included the three pre-fix epochs.
- **Out of scope.** FraGNNet's overfitting work (the dropout ladder) is a different model. More
  data is out of scope by request.

## Ledger

### Unmerged route arms, control `c0`

| arm | run | change | Δ tail vs control | status | audit note |
| --- | --- | --- | ---: | --- | --- |
| `c0_s2` | `1oxhvk0n` | seed 811237 | +0.0042 | seed spread | same code; above `c0` at 56 of 64 epochs with no change |
| `r0` | `qyjlg9tb` | boundary block removed | −0.0129 | fail (diagnostic) | init confounded, [correction 7](#corrections-to-the-record) |
| `r1` | `s9p8neke` | cut term from a 1-layer local GINE | −0.0029 | no benefit | inside the seed spread; clean ablation (forked RNG) |
| `r2` | `9lt5s2u8` | encoder GINE, no global attention | +0.0006 | null | also changes the proposal's encoder and init |
| `k1` | `wg0nxymi` | dropout 0.2 on `CutBondEncoder` hidden and on the token | +0.0074 | **pass, replicated** | `k1_s2` (`e39ko0zr`) +0.0048 vs `c0_s2` |
| `m1` | `liwjgdq6` | 16-dim per-molecule sink from epoch 4 | −0.0203 | fail | mechanism unverified, [correction 10](#corrections-to-the-record) |
| `gm1` | `8ifz8p3t` | 30% of atoms' global message → batch mean | +0.0051 | unresolved pass | no replicate, no gate 2-4 read |
| `gm2` | `msmnydk7` | Gaussian noise 0.5 σ on the global message | −0.0029 | no benefit | |
| `gm3`-`gm6` | `aijxahxb`, `ou0ajppm`, `fr3dcnm1`, `7qw60mn0` | another molecule's message (atom, molecule, feature units; 0.3, 0.6) | +0.0023, +0.0006, −0.0029, −0.0025 | no benefit | all below `gm1` on the same seed; each inside the seed spread |
| `k1_ema` | `j4u25qfd` | `k1` + EMA 0.999 | +0.0029 vs `u2`; +0.0145 vs `c0` | no stacking | sub-additive: (`k1` − `c0`) + (`u2` − `f0`) = +0.0168 |
| `u2_s2` | `qwhsp2fg` | EMA 0.999, seed 811237 | +0.0072 vs `c0_s2` | EMA replicated | |
| `w1` | `tggzldtj` | weight decay 0.1 | +0.0010 | null | reaches every tensor; merged full-length `wd10` never ran |
| `nl1` | `107bmt51` | lighter-side pool only, heavier side as `c_heavy_side` | **+0.0037** | **unresolved, not null** | [correction 1](#corrections-to-the-record) |
| `nl2` | `djhbz6z7` | `nl1` + GINE | −0.0015 (−0.0021 vs `r2`) | mild underfit vs `r2` | [correction 2](#corrections-to-the-record) |
| `nl3` | `kkp8ug11` | `nl2` + `k1` dropouts | −0.0178 vs `nl2`, epochs 21-30 | underfit, crashed at epoch 30 | not OOM, not wall time; no log uploaded |
| `m2` | `41mnwm9f` | `m1` with the sink dropped on half the spectra | +0.0035 (epochs 31-40) | running | +0.0238 over `m1` at the same epochs |
| `s1`, `s1_k1_ema` | none | cosine LR schedule | — | **never submitted** | no W&B run of either name in the project |

### Ranking-transfer arms, control `f0`

| arm | run | change | Δ tail | status | audit note |
| --- | --- | --- | ---: | --- | --- |
| `u2` | `c8hh9j1b` | EMA 0.999 | +0.0094 (+0.0116 vs `c0`) | **pass, replicated** | validation and checkpoints use the average, BatchNorm statistics included |
| `h1` | `rmlyskqj` | allocator dropout 0.1 → 0.2 | −0.0067 | fail | |
| `h2` | `4jmgg2o0` | allocator width 256 → 128 | −0.0015 | null | |
| `u3` | `71xhfgzi` | switch EMA every epoch | −0.0089 vs `u2` | fail (underfit) | |
| `f1` | `6h67zps9` | cleaned training target | +0.0001 val, +0.0007 test (cross-eval) | closed null | |

### Merged pre-plan attempts, control `c4a_base` mean

| arm | run | change | Δ | status | audit note |
| --- | --- | --- | ---: | --- | --- |
| `drop03` | `erfeoywq` | all dropouts 0.3 (sweep, epoch 64) | −0.0149 | underfit | the only gap-narrowing case with train cosine logged |
| `wd10` | `kalc4mzo` | weight decay 0.1 (sweep, epoch 64) | +0.0015 | null | the full-length rerun three docs asked for never ran |
| `layernorm`, `graphnorm` | `i1a4l2ya`, `ucwtbubk` | normalization swaps (sweep) | −0.0024, −0.0025 | null | |
| `c4d_drop02` | `zx7zxqrv` | all dropouts 0.2 | −0.0033 | null | never read before; one of only two Tier D arms that ran |
| `c4d_drop_head03` | `kjkdr22o` | head dropout 0.3 | +0.0019 | null | `c4d_drop03`, `c4d_wd10` and `c4d_drop_mol03` never ran, so the drop03 decomposition is incomplete |
| `d1`, `d2`, `d3` | `oqobubu7`, `2369y1j0`, `rcfzf67p` | cosine LR decay (`d3` with warmup) | +0.0053, +0.0052, **+0.0084** | pass | the docs quote best-so-far (+0.0043, +0.0041, +0.0072) |
| `t2` | `49ub2p8y` | EMA 0.998 | +0.0097 vs `c4a_base_s1` | **pass** | |
| `a1_prenorm` | `q4t4udee` | pre-LN head | +0.0056 | pass | architecture, not aimed at memorization |
| `a2_full_attention` | `cgomm8v4`, `xyessyn1` | full attention, head cut | +0.0030 | null | head cut 43.6% (1,386,763, CPU build); the 40.8% cited is `a6`/`a7`'s |
| `b3_wd_groups` | `gjeqjp44` | norms and biases exempt from decay | −0.0002 | null | [correction 12](#corrections-to-the-record) |
| `e2`, `e3` | `955di4yn`, `588cxux6` | pool atom dropout 0.1, 0.2 | −0.0009, +0.0002 | null | |
| `g1` | `xb9ic2pp` | 5-layer encoder, no bond head | −0.0160 | fail | worse on train and val: an architecture defect |
| `c4a_no_bbe` | `pkavjg8z` | boundary encoder replaced by endpoint mean | −0.0024 | null | |
| `p3` | `w1dj2ahw` | proposal-only weight decay 0.1 | +0.0021 | null | proposal branch only |

### Post-hoc levers, unmerged `f0`/`f1`

These are from the target-cleaning cross-eval (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_target_cleaning_cross_eval_2026_09_23.md`)
and the ranking decomposition (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_ranking_decomposition_2026_09_23.md`), and were not
re-derived here.

| lever | gain |
| --- | ---: |
| ensemble of the two runs | +0.020 on val and test, rising with CE |
| weight average of two checkpoints of one run | +0.0029 / +0.0036 over the best checkpoint |
| predictions raised to the power 0.75 | +0.007 |

### Arms in the same W&B group that are not memorization levers

`n1`-`n4` change only the CE input (`ce_insert_type` or `int_embedder`). `sp0`/`sp1` test the
CE-context sampler and a shared proposal. They are read in the
[QET finding](masster_qet_projection_candidate_union_2026_09_26.md) and the
[shared-proposal audit](masster_shared_proposal_pathway_audit_2026_09_26.md). Their current state
is in corrections 4 and 5.

## Corrections to the record

1. **`nl1` is not null.** It finished at +0.0037, above `c0` at 52 of 64 epochs. Train loss is
   worse by 0.0039, val loss better by 0.0041, and coverage is equal. The docs hold only the
   mid-run reads (+0.0015 projected, +0.0022 at epoch 39). The +0.0037 is above the +0.003 gate
   but inside the 0.0042 seed spread, and the token widening changes initialization. So it is an
   unresolved single-seed pass that needs a second seed.
   - Stale in: the plan banner and 'Batch 3 read', the
     energy-vs-parts (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md`) 'What this changes',
     and MEMORY.md.
2. **`nl2` underfits against its same-code control `r2`.** Its val cosine is −0.0021 against
   `r2`, train loss +0.0095 and val loss +0.0221. That contradicts 'train fit does not move'.
   Against `c0` it is −0.0015.
3. **"The lookup moved to another route" does not follow from `nl1`/`nl2`.** Several inputs stay
   exactly as in `c0`, whichever side the token keeps:
   - the boundary block (bit-identical under the toggle, checked on CPU);
   - `c_mass` and the new `c_heavy_side`, which is close to the heavier side's formula;
   - the candidate set that the allocator attends over. The
     set-message probe (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md`)
     rates that route the most train-specific (3.37x).

   "Rerouted" cannot be told apart from "never removed" from these arms alone. The probe chain
   has since run
   (read (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md`)):
   - `nl1` still writes the key in global attention (2.89x), so the heavier side's removal never
     removed the key.
   - In `r2`, trained without global attention, the lookup did reroute. The set message carries
     almost twice `c0`'s train-specific fit.
4. **`sp0`/`sp1` were resumed, not resubmitted.** Resubmitting under the same job names found
   `job_id/<job>.id`. The runner then resumed the same W&B ids from `last.ckpt`
   ([`runner.py:897-912`](../../../src/fragnnet/runner.py#L897-L912), `resume: must`).
   - Epochs 0-2 are the pre-fix collapse. From epoch 3 the fixed sampler runs, at 312 steps per
     epoch (`c0`: 308).
   - They recovered but stay below `c0`: −0.0221 (`sp0`, epochs 20-29) and −0.0201 (`sp1`,
     28-37).
   - Two things could cause the remaining gap, and nothing separates them: warm-starting from
     three poisoned epochs, and whole-molecule batch composition. A run under a new job name,
     or sharing on `c0`'s random batches (the audit's own suggestion), would.
   - Stale: 'void, resubmit under new run ids' in the audit and in project-state. One auditor
     read them at −0.21 by including epochs 0-2; that read is wrong.
5. **The CE-input arms.**
   - `n2` and `n3` finished at −0.0102 and −0.0085, clustered with `n1` (−0.0121). The docs
     still hold their epoch 6-9 reads, including 'n2 near c0'.
   - `n4` was submitted and is running at +0.0060 over epochs 51-60, better on train and val
     loss. The docs say 'not submitted' and predicted no effect. Its embedder change moves the
     initialization, so read it against the seed spread.
6. **`m2` tracks `c0`, not `m1`.** At epochs 31-40 it is +0.0035 over `c0` and +0.0238 over `m1`.
   The "long shot" framing no longer describes it; read it at epoch 63.
7. **`r0`'s −0.0129 is confounded with initialization.** `CutBondEncoder` is built outside
   `th.random.fork_rng`
   ([`masster_model.py:1720`](../../../src/fragnnet/model/masster_model.py#L1720)). Omitting it
   shifts the RNG stream for every module built after it. In a matched toy build, all of
   `charge_head` and 29 of 59 `intensity_head` tensors differ.
   - The plan's guarantee ('every other initialization exactly as in `c0`') holds for `r1` and
     `m1` only.
   - The direction stands at three times the seed spread.
   - The same caveat applies to `r2` (encoder shape), `nl1`/`nl2` (+19 token columns), `h2` and
     `n4`. The parameter-free arms (`k1`, `gm*`, `w1`, EMA, `m1` with its forked table) keep
     `c0`'s initialization and data order.
8. **The probe's val sample is not random.** The debug probe reads the first 60 of 102 val
   batches through `SequentialSampler`
   ([`runner.py:386`](../../../src/fragnnet/runner.py#L386)), and `val_ids.csv` is 99.5% sorted
   by `spec_id`. Its train sample is random.
   - For all seven checkpoints checked, the probe's val cosine is 0.010-0.014 above the
     full-split W&B value, so absolute gaps are understated by about 0.011.
   - The paired contrasts share batches, but val damage depends on the novelty mix (global
     attention's val damage rises 0.028 → 0.131 with similarity). So the damage ratios may move
     on a random val sample.
   - The route probe (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_memorization_route_probe_2026_09_23.md`) and
     ranking decomposition (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_ranking_decomposition_2026_09_23.md`) describe a CE-context
     sampler and `skip_too_big` that these checkpoints never used (plain `RandomSampler` or
     `SequentialSampler`, batch 512).
9. **The eval-mode gap table overstates `k1` and `r2` slightly, and cannot be reproduced.**
   - `c0` was read at epochs 42 and 63 and the arms at 56-61. `c0`'s own gap grows from 0.1760 to
     0.1834 between those epochs.
   - Interpolated to each arm's epoch, `k1` closes about 0.015 (not 0.017) and `r2` about 0.005
     (not 0.007), inside the instrument's ~0.003 noise.
   - Its raw outputs were in a session scratchpad and are gone.
10. **`m1`'s mechanism is untested.** "The sink absorbed transferable signal" and "the shared
    weights stop getting gradient once the sink fits a spectrum" predict the same logged curves.
    The distinguishing read (eval-mode train cosine with the sink on) was never built. Each
    molecule's sink norm in `saved_ckpts/m1_sink_liwjgdq6.ckpt` is a free per-molecule
    memorization score that nobody has computed.
11. **Merged runs recorded as never run, or never read.**
    - Recorded as never run, but ran and are null: `e6_ring_topology_cut_bond` (`kc04u2cd`,
      +0.0015), `c4a_ring_topology` (`3gqgkl4x`, −0.0027) and `c4a_complement_composition`
      (`lxf97tck`, +0.0012). The claim appears in at least five docs, including the
      [cut-set plan](../../plans/masster/masster_cut_set_features_plan.md) and the architecture
      review todo.
    - Tier D of the 4 h campaign: only `c4d_drop02` and `c4d_drop_head03` ran (−0.0033 and
      +0.0019, never read). `c4d_drop03`, `c4d_wd10` and `c4d_drop_mol03` have no run.
    - The intensity-escape ladder finished null: `e0` 0.7231 against `e1` 0.7230 over two seeds.
      That final read was not written up.
    - Other arms recorded as unrun that ran:
      - FraGNNet `neutral_loss_formula` (`b7witrjj`, 2026-07-15, crashed at epoch 63) and
        `graph_feature_parity` (`m8qgi0ft`, 2026-09-23, crashed at epoch 77). Neither has been
        read.
      - The MSGYM GLACIER-parity arm `ll04i00m`, read null elsewhere, which one literature
        note still listed as unsubmitted.
12. **`b3_wd_groups` and `e2` have no train cosine.** The plan says they narrowed the gap by
    lowering train fit, but both ran with `compute_train_auxiliary_metrics: false`. That leaves
    only a train-mode primary-loss gap.
    - `b3`'s change (0.016) is 1.1-1.2 times the family's own floor for that statistic.
    - `e2`'s (0.047) clears it.
    - `drop03` is the only one with train cosine.
13. **Verdict wording.** 'r1 fails', 'gm2 fails' and 'the swap family is closed' rest on
    single-seed deltas inside the seed spread. The ordering is solid: every swap arm sits below
    `gm1` on the same seed. Read these as "no detectable benefit", which is a resource decision,
    not a measured harm.
14. **Minor.**
    - The ranking-transfer plan calls `f0` an 'exact same-code control'. It predates the
      `CutBondEncoder` dedup, which is equal up to float rounding.
    - The `a2` head cut is 43.6% (1,386,763 parameters; `a3` is `a2` plus 45). The 40.8% in the
      memorization plan and the epoch-128 finding belongs to `a6`/`a7`.

## Noise floor and the gate

| pair | difference | what it measures |
| --- | ---: | --- |
| `c0_s2` − `c0` | +0.0042 | seed, same code |
| `c0` − `f0` | −0.0022 | same seed, `CutBondEncoder` dedup (float rounding) |
| `k1` effect at the two seeds | +0.0074, +0.0048 | arm effect across seeds |
| EMA effect at the two seeds | +0.0116, +0.0072 | arm effect across seeds |
| merged `c4a_base_s1` − `s2` | 0.0007 | merged seed spread |

- **A single unmerged arm delta has a standard error of about 0.003 (0.002-0.004).** The +0.003
  gate is one standard error, so a null arm clears it 9-24% of the time.
- **Chance passes.** Fourteen single-change memorization arms ran against `c0` on seed 397623,
  which predicts 1.3-3.4 chance passes. Three passed: `k1`, `gm1` and `nl1`. `k1` held on the
  second seed (mean +0.0061, about 3 standard errors). `gm1` and `nl1` are unresolved.
- **"Epochs above control" is not independent evidence.** `c0_s2`, which changes nothing but the
  seed, sits above `c0` at 56 of 64 epochs.
- **Parameter-free arms share `c0`'s initialization and data order.** Their deltas are paired
  and tighter than the seed spread; `c0` − `f0` (0.0022) is the closest estimate.
- **Structural arms are closer to a seed change** ([correction 7](#corrections-to-the-record)).
- **The merged screen is about five times tighter**, which is why its ±0.005 gate resolved
  more.
- **Proposed reading rule:** a single seed must reach about +0.006 (two standard errors) or get
  a second seed; −0.003 means "no detectable benefit".

## What the attempts show together

- **The gap has not moved.** It is about 0.18 on the probe's (easier) val sample in every arm
  read: `c0` 0.1834, `k1` 0.1665, `r2` 0.1765, `r1` 0.1790, `r0` 0.1846, `u2` 0.1808. `m1` reaches
  0.0919 only by losing train fit.
- **The gap is not replicate noise being fit.** Two spectra of one train molecule at the same NCE
  agree at 0.946 (1,091 pairs). Train fit (0.924) sits near that ceiling and val (0.744) far below
  it.
- **Removing an identity route has not helped once.**
  - Boundary block: `r0` −0.0129, `c4a_no_bbe` −0.0024.
  - Global attention: `r2` +0.0006.
  - Heavier side plus global attention: `nl2` −0.0021 vs `r2`.
  - Allocator capacity: `h2` −0.0015; `a2` +0.0030 inside ±0.005.
  - `g1` −0.016.

  The lookup either re-forms or was never confined to the route removed.
- **Noise helps only while other routes remain.** `k1` gains on `c0`, replicated, and `gm1` is a
  possible gain. The same `k1` noise costs −0.018 on `nl2`, which has already lost two routes.
  Per-feature swaps, 0.6-rate swaps, Gaussian message noise and allocator dropout (`h1`) hurt or
  do nothing.
- **Variance reduction is the largest lever, and it does not narrow the gap.**
  - EMA: +0.009 to +0.012 on both lineages.
  - Two-run ensemble: +0.020.
  - Merged LR decay: +0.005 to +0.008.

  All three lift train and val together. The ensemble gain says two runs disagree most on novel
  molecules and at high CE (variance-novelty (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_variance_novelty_2026_09_24.md`)).
- **The route map describes the trained checkpoint, not causal routes.**
  - Its top route (global attention, 2.6x) did not pay when removed (`r2`).
  - It predicted `k1` would be weak, and `k1` is the one replicated structural gain.
  - It may point to where training noise helps (`gm1`), but that is unreplicated.
- **Stacking has not added up.** `k1` + EMA gives +0.0145 over `c0`, against +0.0168 for the
  two effects summed ((`k1` − `c0`) + (`u2` − `f0`)).
  On the merged lineage, `a7` fell below `a1` and `a6`, and `e5` below `e4`.

## Untried levers, by evidence over cost

| rank | lever | state | why |
| ---: | --- | --- | --- |
| 1 | `s1`, cosine LR on unmerged | built and smoke-tested; never submitted | merged `d3` +0.0084; config only. It may overlap EMA's gain. |
| 2 | second seeds of `gm1` and `nl1`; read `m2` and `n4` at epoch 63 | configs to write (seed 811237, against `c0_s2`) | three unresolved single-seed positives |
| 3 | `m1` sink norm as a per-molecule memorization score | checkpoint local; CPU only | unblocks the memorization score and low-loss exclusion |
| 4 | gates 2-4 on `k1`, `gm1`, `u2`, `k1_ema` | checkpoints uploaded | no arm has had them; the running chain covers `r2`, `nl1` and `nl2` |
| 5 | set-message substitution in the allocator (`gm1`'s form on the 3.37x route) | **built 2026-09-27 as `sm1`**, with the key-mask companion `sd1`; not submitted | the most train-specific route on f0 and `c0`, and where the lookup went in `r2` |
| 6 | Goldfish masked peaks; label smoothing toward the formula prior | no code | cheap and independent of the route |
| 7 | `t1`/`u1` block-level boundary dropout | code exists, never run | dropped on an argument; `k1`'s hidden-layer dropout covers part of it |
| 8 | a bottleneck on the token or global message (`v1`/`2c`), gradient agreement (`c1`), late SAM, shard soups, out-of-fold distillation | no code | heavier; route-agnostic caps |
| 9 | collision census (Step 0.4), then the tied cut term (2a); identity-blind lability prior (2A) | not built | 2a needs the census; 2A's gate is in the grey zone |

Not memorization levers but ready and unrun: the architecture arms `a4`, `a5` and `b4`.

## Next checks

1. **Done 2026-09-27:** the stale records above are corrected in place, with pointers here. That
   covers:
   - the memorization-route and ranking-transfer plans;
   - the route probe, route map, energy-vs-parts and ranking-decomposition findings;
   - the shared-proposal audit and the QET finding;
   - the cut-set, architecture-review, campaign and escape plans, and the epoch-128 finding;
   - the feature-information audit, the feature-set reference, the GLACIER and
     displacement-literature notes and the identity-supervision finding;
   - project-state and the README rows.
2. Resubmit `sp0`/`sp1` under new job names, or test sharing on `c0`'s random batches.
3. Replicate `gm1` and `nl1` at seed 811237 against `c0_s2`.
4. Submit `s1`.
5. Run one probe condition on the full val split (`--batch_cutoff -1`) to size the bias of the
   `spec_id`-ordered sample.
   - **Done 2026-09-28, on `i1`**
     ([read](masster_i1_memorization_read_2026_09_28.md#the-probes-val-sample)).
   - The prefix is 90% Elite (the split is 74%) and less novel. That instrument mix is most of the
     bias: Elite prefix spectra read as on the full split, while Lumos prefix spectra are easier.
