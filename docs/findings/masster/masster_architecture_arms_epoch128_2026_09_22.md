# MASSter architecture arms, read at the matched 128-epoch screen, 2026-09-22

> **Status:** FINDING · **Updated:** 2026-09-27
> 128-epoch architecture-arm read against the 0.7270 control (gate +-0.005): `a1_prenorm` +0.0056 passes; `b2` -0.0062 and `g1` -0.0160 are rejections; sections 3-5 add the reproducibility floor, the full-attention re-check and `g1`.
>
> **§5 (added 2026-09-23) reads `g1`, one 5-layer GINE_GSA encoder with no bond-refinement
> head and no `CutBondEncoder`. It is a REJECTION: −0.0160 against the control, −0.0136 against
> the two-key `c4a_no_bbe` contrast, and below both controls at every epoch. The loss is
> intensity fit on both splits, not the proposal.**
> **§4 (added 2026-09-23) re-checks every full-attention run from raw history and reads the
> 630-epoch arms past 127 against the horizon probe: full attention alone is +0.0017 to +0.0030
> depending on the control, and `a1`/`a6` settle 0.006–0.008 above it by epoch 200.**
> Three reads at the same screen against the same control. **§3 (added last) reads
> `a2_full_attention` and `b1_bond_sum`, which resolves §1's capacity confound — `a6_ce_film`'s
> +0.0063 is half topology and half CE-FiLM, and the pairwise mass bias is an exact null at
> matched topology — and §3.1 turns the accidental duplicate submission into the family's first
> measured run-to-run reproducibility floor: 0.0004 on the tail mean, 0.0025 on a single epoch,
> and 0.011–0.014 on the loss-drift columns §1 reads mechanistically.** **§2 (added later the same day) reads
> the five candidate-token feature arms and the intensity-escape pair, which the plans still
> recorded as "nothing trained" — they had in fact launched, and all six are nulls.**
> §1 closes §1 of the [architecture review todo](../../plans/masster/masster_architecture_review_todo.md):
> the six arms that ran in the 2026-09-21 campaign and were killed at the wall clock between
> epochs 179 and 210 are now read against both `c4a_base` replicates at matched epoch.
> **Two arms clear the replicate floor (`a1_prenorm` +0.0056, `a6_ce_film` +0.0063), one is a
> rejection (`b2_identity_residual` −0.0062), one is an exact null (`b3_wd_groups` −0.0002).**
> Coverage does not move in any of the six — `val_wrecall` agrees to **±0.0006** across all
> eight runs — so every difference below is intensity allocation, not candidate construction.
> Read-only W&B queries; no code, config or run changed.

## 1. The six architecture arms

### Method

Runs pulled from `frag-gnn/frag-gnn` on 2026-09-22 with `scan_history(keys=None)` — the full
row stream, not the 500-point sample `run.history()` returns — then forward-filled per `epoch`,
because train and validation metrics land on alternating steps. This is the trap recorded in the
[campaign read-out](masster_campaign_readout_2026_09_21.md) and the
[overfit read-out](masster_overfit_readout_2026_09_22.md).

Metric is `val_cos_sim_0.01_epoch/mean`. Two statistics per arm, because best-so-far is biased
upward by single-epoch noise and every arm here carries a different number of draws:

- **tail mean**, epochs 118–127 (n=10) — the primary read.
- **best-so-far ≤ epoch 127** — comparable to the numbers already in the run index.

Within-run epoch-to-epoch spread over epochs 100–127 is sd **0.0010–0.0030** (mean |Δ| 0.0011–
0.0019), so a 10-epoch tail mean has a standard error near 0.0005 and the two statistics should,
and do, agree to ~0.001 except where an arm is still climbing.

### The 630-vs-128 horizon does not confound this read

Every one of the eight runs has **`lr_schedule: False`** and a constant `lr: 1e-3`. The
`lr_schedule_type: exponential` / `lr_warmup_steps: 1000` keys in their configs are inert — the
same "`lr_schedule` is `False` in 83/83 runs" fact the campaign read-out recorded. Nothing in the
optimizer reads `max_epochs`, so the six arms' `min_epochs: 378 / max_epochs: 630` changed only
*when the job would have stopped*, never the trajectory. Epoch 127 of an arm and epoch 127 of a
control are the same amount of the same optimization.

### Two metadata traps, both harmless here

- All six arms are tagged `vulcan_l40s` and grouped `nist20_mass_shatter_l40s`. **They ran on Fir
  H100 80GB**, the same hardware as both controls, within 35 minutes of them. The tag is wrong;
  `run.metadata['gpu']` is the evidence. Same class as the `train_aux_metrics` tag the overfit
  read-out caught on `c4a_base_s1/s2`.
- All six carry **seed 397623**, which is `c4a_base_s1`'s seed. So each arm is a seed-matched
  paired comparison against `s1` specifically; `s2` (seed 811237) only supplies the replicate
  spread. The paired delta and the delta to the control mean agree to 0.0004 throughout.

### Result

Control: tail mean **0.7270** (s1 0.7266, s2 0.7273, spread **0.0007**); best ≤127 **0.7286**
(0.7282 / 0.7289). Gate is ±0.005 against the family's 0.003–0.005 replicate floor.

| arm | run | key(s) vs base | tail 118–127 | Δ control | Δ paired (s1) | best ≤127 | Δ best | verdict |
|---|---|---|---|---|---|---|---|---|
| `a6_ce_film` | `0gbdejd8` | `intensity_ce_film` **+ `attention: full`** | 0.7332 | **+0.0063** | +0.0066 | 0.7344 | +0.0058 | **PASS**, confounded |
| `a1_prenorm` | `q4t4udee` | `intensity_norm_placement: pre` | 0.7326 | **+0.0056** | +0.0059 | 0.7333 | +0.0048 | **PASS**, clean |
| `a7_ce_both` | `zgm04inn` | ce_film + ce_pair + mass_pair + pre-LN **+ full** | 0.7304 | +0.0035 | +0.0038 | 0.7331 | +0.0046 | hold |
| `a3_full_massbias` | `4fg0x5vw` | `intensity_mass_pair_bias` **+ `attention: full`** | 0.7297 | +0.0028 | +0.0031 | 0.7324 | +0.0038 | hold |
| `b3_wd_groups` | `gjeqjp44` | `weight_decay_exclude_norm_bias` | 0.7268 | −0.0002 | +0.0002 | 0.7285 | −0.0001 | **null** |
| `b2_identity_residual` | `icii1zho` | `gnn_residual_activation: identity` | 0.7207 | **−0.0062** | −0.0059 | 0.7247 | −0.0039 | **REJECT** |

The ordering is stable: recomputing best-so-far at cuts 96, 112, 120 and 127 never reorders
`a6 ≳ a1 ≈ a7 ≳ a3 > b3 > b2`, and the three tail windows (118–127, 108–127, 88–127) move every
delta by at most 0.0008.

**`a1_prenorm` is truncation-limited at the screen**: its best ≤127 lands *on* epoch 127, and its
128–135 values (0.7336, 0.7325, 0.7333, 0.7305, 0.7330, 0.7331, 0.7326, 0.7329) show it flat
rather than rising, so +0.0056 is the right number, not a floor.

### The two PASSes are not the same quality of evidence

`a1_prenorm` is parameter-identical to the base (9,215,573) and differs by **one key**. It is the
clean result.

`a6_ce_film` also sets `attention_type: full`, which deletes the induced layer's latent attention
block, feed-forward and inducing tokens — a **40.8% cut to the intensity head**
(2,457,355 → 1,454,347). So `a6` is "CE-FiLM, plus full attention, plus a 41% smaller allocator"
and is not attributable until `a2_full_attention` supplies the topology-only control. What it does
establish, without needing that control, is that **the campaign's best number is reachable while
running 41% less of the module the campaign blames for the ceiling.**

### Three contrasts that are confound-free right now

`a3`, `a6` and `a7` all carry the identical attention change and within 0.1% the same parameter
count, so differences *among them* isolate the bias terms:

| contrast | Δ tail | reading |
|---|---|---|
| `a6` − `a3` | **+0.0035** | CE-FiLM beats the pairwise mass bias at matched capacity and matched topology |
| `a7` − `a6` | **−0.0028** | adding mass-pair + CE-pair bias + pre-LN *on top of* CE-FiLM costs |
| `a7` − `a3` | +0.0007 | null |

Both non-null contrasts sit at the replicate floor, not above it, and each is n=1. But they point
the same way and they agree with the arm-vs-control column: the effects are **not additive**.
`a1` (+0.0056) and `a6` (+0.0063) individually beat `a7` (+0.0035), which contains both.

### `a1_prenorm` also halves the calibration drift — the strongest mechanistic signal in the set

§3 of the [overfit read-out](masster_overfit_readout_2026_09_22.md) measures each run's
`val_primary_loss_epoch/mean` against its own optimum. At matched epoch 127:

| run | best primary-loss epoch | loss at 127 | drift from own optimum | train/val loss gap at 127 |
|---|---|---|---|---|
| `c4a_base_s1` | 52 | −2.7182 | +0.0380 | 0.6320 |
| `c4a_base_s2` | 66 | −2.7266 | +0.0255 | 0.6171 |
| **`a1_prenorm`** | **120** | −2.7665 | **+0.0121** | **0.5924** |
| `b3_wd_groups` | 64 | −2.7299 | +0.0204 | 0.6086 |
| `a3_full_massbias` | 80 | −2.7263 | +0.0362 | 0.5931 |
| `a6_ce_film` | 78 | −2.7479 | +0.0386 | 0.5962 |
| `a7_ce_both` | 111 | −2.7349 | +0.0450 | 0.6042 |
| `b2_identity_residual` | 82 | −2.6829 | +0.0431 | 0.6380 |

Pre-LN **delays the calibration turn by ~60 epochs** (120 vs 52/66) and cuts the post-turn drift
**2.6x** (+0.0121 vs a +0.0318 control mean), while also giving the best validation primary loss
of the eight. That is the behavior the review predicted for post-LN under a constant unwarmed LR,
measured. Note the turn is not abolished, and note `a7` has pre-LN too and the *worst* drift of
the eight — under full attention the effect reverses.

`b3_wd_groups` is the second-cleanest case of the same shape: an exact cosine null that still
narrows both the drift (+0.0204) and the loss gap (0.6086). Excluding norms, biases and the
inducing tokens from weight decay changes the optimization and does not change what ships.

### Coverage moves in none of them

`val_wrecall`, mean of epochs 118–127:

| | s1 | s2 | a1 | a3 | a6 | a7 | b2 | b3 |
|---|---|---|---|---|---|---|---|---|
| `val_wrecall` | 0.8911 | 0.8912 | 0.8917 | 0.8914 | 0.8912 | 0.8912 | 0.8912 | 0.8913 |

Spread **0.0006** across eight runs whose cosine spans 0.0125. `val_recall` is likewise flat
(0.6818–0.6826). This is an eight-run confirmation of the overfit read-out's §4.1 conclusion on
a population it did not cover: at this operating point **candidate construction is fixed and
architecture changes act only on allocation.** `val_wprecision` does move (`a1` −0.0111,
`a6` +0.0035, `b2` −0.0057), and it moves *against* cosine for `a1` — the best arm proposes its
retained mass slightly worse while ranking it better.

### Consequences for the review items

| # | Review item | Arm | Verdict |
|---|---|---|---|
| 1 | Candidate head is mass-blind | `a3` | **hold** (+0.0028, and confounded by the head cut) |
| 2 | Post-LN under constant unwarmed LR | `a1` | **PASS, clean** (+0.0056 cosine, drift 2.6x lower, turn 60 epochs later) |
| 3 | GNN residual branch is ReLU-gated | `b2` | **REJECT** (−0.0062). Keep the ReLU gate; do not re-run. |
| 7 | Weight decay on norms, biases, latents | `b3` | **null** (−0.0002). Hold, not rejected — it moves the loss, not the metric. |

Items 4 (RWSE), 5 (`b1_bond_sum`) and 6 (`a2_full_attention`) are untouched by this read and
stay open.

**What this changes about the submission order.** `a2_full_attention` was promoted to co-lead
because three ran arms could not be attributed without it. That argument is now sharper, not
weaker: `a6` is the campaign's best arm *and* its second-best clean candidate is a different
mechanism entirely, so `a2` decides whether the CE-FiLM result is a bias result or a topology
result. `b1_bond_sum` is unaffected. `a4` and `a5` were conditional on §1 and their condition is
met — `a3` did not clear the floor, but `a1` did, and `a4` stacks pre-LN onto the mass bias.

**What it changes about promotion.** `a1_prenorm` is the first clean, parameter-identical,
one-key architecture arm in this family to clear the floor. It is still n=1 against an n=2 control
whose 0.0007 spread is tighter than the family's history, which is the same caveat the `d3` LR
result carries. Both `a1` and `d3` touch optimization under a constant LR and neither has been run
with the other; they are the two candidates worth replicating and then combining.


## 2. The candidate-token feature arms and the intensity escape — added 2026-09-22

**These runs existed and the plans did not know it.** The
[cut-set plan](../../plans/masster/masster_cut_set_features_plan.md) banner said "five matched arms built,
nothing trained" and the project-state entry for the candidate-token work said "No production run
launched". Both were written before the arms were submitted at 19:57–20:01 UTC on 2026-09-22. Five
finished at 127 epochs, on the same screen and against the same `c4a_base` control as §1. Read the
same way: tail mean of epochs 118–127, control **0.7270**, gate ±0.005.

| arm | run | key(s) | tail 118–127 | Δ control | best ≤127 | verdict |
|---|---|---|---|---|---|---|
| `e4_cut_adjacency` | `kykqk4hr` | `candidate_cut_adjacency_features` | 0.7301 | **+0.0032** | 0.7325 | hold |
| `e5_cut_bond_plus_adjacency` | `udz5jwfx` | adjacency **+** cut-bond | 0.7291 | +0.0021 | 0.7311 | hold |
| `e3_cc_atom_drop02` | `588cxux6` | `candidate_atom_dropout: 0.2` | 0.7271 | +0.0002 | 0.7288 | hold |
| `e2_cc_atom_drop01` | `955di4yn` | `candidate_atom_dropout: 0.1` | 0.7260 | −0.0009 | 0.7299 | hold |
| `e1_cut_bond_feats` | `fj5cfyly` | `candidate_cut_bond_features` | 0.7260 | −0.0010 | 0.7282 | hold |

**No arm clears the floor in either direction.**

*Corrected 2026-09-27:* `e6_ring_topology_cut_bond` did run, though this line said it was never
submitted. It finished 128 epochs on 2026-09-23 with `c4a_ring_topology` and
`c4a_complement_composition`, and all three are null:

- `e6` (`kc04u2cd`): +0.0015;
- `c4a_ring_topology` (`3gqgkl4x`): −0.0027;
- `c4a_complement_composition` (`lxf97tck`): +0.0012.

[Audit](masster_memorization_attempts_audit_2026_09_27.md).

Three readings:

- **Adjacency is the only one with a sign, and it is inside the floor.** That is the direction the
  descriptor measurement predicted — given the pooled cut-bond sum, the ring-sharing flag has 0.000
  residual ambiguity while adjacency has 0.489 — so the mechanism argument survives and the effect
  does not. `e1` (the cut-bond block alone, the larger of the two features at 18 columns) is the
  flattest arm in the set at −0.0010.
- **Stacking is again non-additive and again costs.** `e5` = `e1` + `e4` scores 0.0011 *below* `e4`
  alone. §1 found the same shape on `a7` vs `a1`/`a6`. Two independent families now say that in
  this model a stacked arm is not a bet on the sum of its parts.
- **The atom-dropout arms behave exactly like a regularizer that does not pay.** `e2` has the
  narrowest train/validation primary-loss gap of the seven runs (0.5719 against a 0.6191 control
  mean) and the *lowest* cosine of the five. That is the third instance of the pattern in this
  file: `b3_wd_groups` and `drop03` did the same thing. Narrowing the gap is still not the
  objective.

Coverage is flat here too: `val_wrecall` 0.8913–0.8917 against a 0.8911/0.8912 control, spread
0.0006 over the whole seven-run set.

### The arms are functionally one-key, but not literally

Each e-arm's resolved config differs from `c4a_base_s1` by **eight** keys, not one. Seven are keys
that did not exist when the control ran and resolve to `False`/`0` — `candidate_composition_features`,
`intensity_escape`, `spectral_ce_auxiliary_weight`, `spectral_leak_loss_weight` and the two sibling
candidate keys each arm does not set — plus three inert `spec_params` additions from the
above-precursor filter (`filter_above_precursor_peaks: False`, `above_precursor_ppm: 10`,
`above_precursor_use_instrument_tolerance: True`). All are no-ops at those values, so the
comparison holds, and all five carry `c4a_base_s1`'s seed 397623, making each a paired read.
The one thing worth checking before the next such arm is whether `spec_params` participates in any
dataset cache identity — new inert keys in a hashed dict are how a "one-key" arm quietly changes
its inputs.

### Intensity escape: an exact null at n=2 vs n=2

`e0_independent_s1/s2` (`518x61y4`, `nb9dk9hw`) against `e1_escape_s1/s2` (`magizakg`, `3mnclnx8`),
on the `batch256x2_lb32` base — **not** comparable to the 0.7270 merged control above, but `e0` is
their own matched control. Still running; read at their common epoch 118, tail 109–118:

| | s1 | s2 | mean | seed spread |
|---|---|---|---|---|
| `e0_independent` | 0.7225 | 0.7226 | **0.7225** | 0.0001 |
| `e1_escape` | 0.7224 | 0.7227 | **0.7225** | 0.0003 |

**Δ +0.0000.** This is the tightest null in the family — two seeds a side, spreads of 0.0001 and
0.0003, and the arm means agreeing to four decimal places. Re-read at completion, but nothing about
the trajectory suggests a late separation.

### Two arms from §2 of the review todo are now running

`b1_bond_sum` (`3bqnobq1`) and `a2_full_attention` launched 21:56 UTC at 128/128 epochs, seed
397623 — the lead pair §1 called for. They were at epoch 20 when this was written; do not read
them yet. **They finished; §3 reads them.**

**`a2_full_attention` was submitted twice at the same seed** (`xyessyn1` on `fc11018`, `cgomm8v4`
on `fc10906`). That is a duplicate, not a replicate, and it costs a box that `a5_ce_pair_bias` or
an `a1_prenorm` replicate could have used. It does buy one thing by accident: two identical-seed
runs on different hosts measure the true run-to-run reproducibility floor, which is worth having
given that the 0.0007 control spread is doing a lot of work in every verdict in this file.
**§3.1 is that measurement.**

## 3. `a2_full_attention` and `b1_bond_sum`, read at the same screen

Both finished at 127 epochs, seed 397623, same control, same statistic (tail mean 118–127,
control **0.7270**, gate ±0.005).

| arm | run | key(s) vs base | tail 118–127 | Δ control | best ≤127 | `val_wrecall` | verdict |
|---|---|---|---|---|---|---|---|
| `a2_full_attention` | `xyessyn1` | `intensity_transformer_attention: full` | 0.7301 | +0.0031 | 0.7323 | 0.8911 | hold |
| `a2_full_attention` | `cgomm8v4` | *(duplicate, same seed)* | 0.7298 | +0.0028 | 0.7323 | 0.8914 | hold |
| `b1_bond_sum` | `3bqnobq1` | `bond_endpoint_aggregation: sum` | 0.7256 | −0.0013 | 0.7279 | 0.8916 | **null** |

`a2` mean tail **0.7300**, Δ control **+0.0030**. Coverage is flat again — `val_wrecall` 0.8911–
0.8916 against the 0.8911/0.8912 control, so the eight-run spread of 0.0006 in §1 now holds over
eleven runs.

### §1's confound resolves: the topology change carries half of `a6`

`a2` is the topology-only control §1 asked for — `attention_type: full` alone, nothing else. Its
intensity head is 43.6% smaller (1,386,763 parameters: `a3`'s 1,386,808 less the mass bias). The
40.8% cut belongs to `a6`/`a7`, whose CE-FiLM adds parameters back (*corrected 2026-09-27*;
this line said 40.8%). Against it, the three bias arms that share its topology:

| contrast | Δ tail | reading |
|---|---|---|
| `a2` − control | **+0.0030** | **`attention_type: full` is worth this much on its own, at 44% fewer intensity-head parameters** |
| `a6` − `a2` | **+0.0033** | CE-FiLM's own contribution, at matched topology and capacity |
| `a3` − `a2` | **−0.0002** | **the pairwise mass bias is an exact null** — all of `a3`'s +0.0028 was the topology |
| `a7` − `a2` | +0.0005 | stacking mass-pair + CE-pair + pre-LN on full attention adds nothing |

So `a6_ce_film`'s +0.0063 splits about evenly: **+0.0030 topology, +0.0033 CE-FiLM.** Neither half
clears the ±0.005 gate alone; the sum does. `a1_prenorm` (+0.0056 from one key, no capacity change)
remains the only clean single-key PASS in the file, and is now also the cheaper of the two.

The `a3` result is the sharper one. §1 recorded `a3_full_massbias` at +0.0028 and held it; the
topology control says the mass-pair bias contributes **−0.0002**. The architecture review's
mass-blindness item is not answered by this term — what moved was deleting the induced layer.

### 3.1 The reproducibility floor, measured

`xyessyn1` and `cgomm8v4` are the same config and the **same seed 397623** on different hosts.
`deterministic: False` in both configs, so the `deterministic` W&B *tag* these runs carry is
wrong. They diverge from **epoch 0** (+0.0060) and never reconverge:

| statistic | `xyessyn1` | `cgomm8v4` | Δ |
|---|---|---|---|
| single epoch, mean \|Δ\| over all 128 | — | — | **0.0025** (sd 0.0029, max 0.0085) |
| single epoch, mean \|Δ\| over ep ≥ 100 | — | — | 0.0023 (max 0.0064) |
| `val_cos` @ epoch 64 | 0.7211 | 0.7264 | 0.0053 |
| `val_cos` @ epoch 100 | 0.7298 | 0.7310 | 0.0012 |
| **tail mean 118–127** | 0.73013 | 0.72977 | **0.00036** |
| best ≤ 127 | 0.73230 | 0.73231 | 0.00001 |

**This validates the statistic this file already uses and invalidates single-epoch reads.** The
tail mean's run-to-run floor (0.0004) is *tighter* than the cross-seed control spread (0.0007),
so the 118–127 window is doing what §1 claimed; a single epoch carries 0.0025–0.0053, which is
larger than five of the eleven arm effects in this file. One pair is n=1 on the floor itself, so
read 0.0004 as an estimate of precision, not a licence to promote 0.001 differences.

### 3.2 The loss-drift column in §1 has a much coarser floor than its cosine column

The same pair, on §1's mechanistic table:

| | `xyessyn1` | `cgomm8v4` | Δ | control pair (s1/s2) |
|---|---|---|---|---|
| best primary-loss epoch | 78 | 78 | **0** | 52 / 66 |
| `val_primary_loss` @ 127 | −2.7344 | −2.7523 | 0.0179 | 0.0084 |
| drift from own optimum | +0.0353 | +0.0241 | **0.0112** | 0.0125 |
| train/val loss gap @ 127 | 0.5984 | 0.5847 | **0.0137** | 0.0149 |

Two consequences for §1:

- **The turn epoch is the reliable part.** Identical seeds give an identical turn (78 vs 78) while
  different seeds give 52 vs 66 — a ~14-epoch seed spread. `a1_prenorm`'s turn at **epoch 120** is
  far outside that, so "pre-LN delays the calibration turn by ~60 epochs" **stands**.
- **The drift magnitude and the loss gap do not support fine ordering.** Their run-to-run floors
  are 0.011 and 0.014. §1's "2.6x cut in post-turn drift" for `a1` (+0.0121 vs a +0.0318 control
  mean) is about 1.7 floor-widths and should be read as *directional*, not as a ratio; and the
  spread among `a3` (+0.0362), `a6` (+0.0386), `b2` (+0.0431) and `a7` (+0.0450) is inside it.
  The same applies to §2's claim that `e2` has "the narrowest train/validation gap" — 0.5719 vs a
  0.6191 control mean is 3.4 floor-widths and survives, but the seven-run *ordering* does not.

## 4. Full attention re-checked, and read past epoch 127 — added 2026-09-23

Recomputed from raw W&B history (`scan_history`, last value per epoch), not from the tables
above. Five runs have ever trained with `masster_intensity_transformer_attention: full`, all
on this lineage and seed 397623: `a2` twice, `a3`, `a6`, `a7`. The older merged-track A/B config
(`config/masster/merged/intensity_attention/..._full_attention.yml`) was never run. Resolved
config against `c4a_base_s1`: `a2` differs only in the attention key (plus ten inert
new-default keys); `a3`/`a6`/`a7` bundle it with other changes, as §1 says. The §1/§3 tail means
reproduce to 1e-4.

**Cost.** Full attention is not slower: 0.817 min/epoch (`a2`) against 0.828–0.838 for the two
controls, at 41% fewer intensity-head parameters. The pairwise-bias arms (`a3`, `a7`) cost
0.965–0.970.

**Past 127 against the horizon probe.** `a1`, `a3`, `a6` and `a7` carried 630-epoch configs and
ran to epochs 179–209 before they were killed; both `c4a` controls stop at 127. The only
seed-matched control past 127 is `tkkesus9` (`long_control`), whose resolved config equals
`c4a_base_s1` except for the epoch horizon (the eight keys it lacks were added later and default
to the control's values). It is **not** an exact control: it ran on earlier code (0.99 vs 0.83
min/epoch) and its 118–127 tail mean is 0.7283, **0.0017 above same-seed `c4a_s1`**, about four
times §3.1's same-code floor. Epoch-matched window means, Δ against `tkkesus9`:

| window | `tkkesus9` | `a1` pre-LN (induced) | `a3` full + mass bias | `a6` full + CE-FiLM | `a7` full + all |
| --- | ---: | ---: | ---: | ---: | ---: |
| 118–127 | 0.7283 | +0.0043 | +0.0015 | +0.0050 | +0.0022 |
| 140–149 | 0.7270 | +0.0063 | +0.0034 | +0.0065 | +0.0050 |
| 160–169 | 0.7277 | +0.0057 | +0.0045 | +0.0055 | +0.0047 |
| 170–178 | 0.7275 | +0.0063 | +0.0045 | +0.0061 | +0.0054 |
| 190–199 | 0.7278 | +0.0063 | — | +0.0068 | — |
| 200–204 | 0.7278 | +0.0077 | — | +0.0070 | — |

**What this adds.**

1. **Topology alone stays below the gate, and its size depends on the control.** `a2` is
   +0.0030 against the `c4a` mean but +0.0017 against `tkkesus9`. No full-attention-only run
   exists past 127.
2. **The §1/§3 orderings hold past the screen.** The control is flat at ~0.7275 from 118 to
   204, while `a1` and `a6` keep climbing and settle 0.006–0.008 above it. `a3` (the null mass
   bias on top of full attention) trails `a6` throughout, and `a7` trails both `a1` and `a6`
   although it contains both of their keys. That is the stacking cost §1 reported, now across
   80 more epochs.
3. **Untested:** pre-LN + full attention alone. `a7` is the only run combining them, and it also
   carries CE-FiLM and both pairwise biases.

## 5. Encoder simplification arm `g1` (added 2026-09-23)

Run `xb9ic2pp` from
`config/masster/ma_mi_merged/encoder_simplify_2026_09_23/g1_gine_gsa5_no_bond_head.yml`:
seed 397623 (paired with `c4a_base_s1`), Fir H100 MIG, 128 epochs, finished. Same method as §1
(`scan_history`, last value per epoch). Resolved against `c4a_base_s1`, it differs by three
keys:

- `mol_num_layers` 2 → 5.
- `masster_bond_num_layers` 3 → 0, which swaps `BondFocusedBreakHead` for
  `EndpointBondBreakHead`.
- `masster_boundary_bond_encoder` off.

Against `c4a_no_bbe` (`pkavjg8z`, same seed) only the first two differ. The two keys missing
from `pkavjg8z`'s W&B config (`masster_fragment_selection`,
`masster_selected_fragment_target`) were added later, and their defaults are the only
behavior that existed then. So `g1` − `no_bbe` is a paired two-key contrast. Parameters fall
from 9,217,181 to 7,694,493: encoder +2,665,728, bond head −3,401,088, `CutBondEncoder`
−787,328. The intensity and charge heads are unchanged.

Tail means over epochs 118–127 unless the row says otherwise:

| | `g1` | `c4a_no_bbe` | `c4a_base_s1` | `c4a_base_s2` |
| --- | ---: | ---: | ---: | ---: |
| val cosine | **0.7110** | 0.7246 | 0.7266 | 0.7273 |
| best ≤ 127 | 0.7160 @ 111 | 0.7263 @ 123 | 0.7282 @ 122 | 0.7289 @ 123 |
| val cosine @ 32 / 64 / 100 | 0.6846 / 0.7057 / 0.7039 | 0.7111 / 0.7208 / 0.7232 | 0.7168 / 0.7202 / 0.7258 | 0.7162 / 0.7195 / 0.7272 |
| val primary loss | −2.6506 | −2.6969 | −2.7251 | −2.7264 |
| train primary loss | −3.2310 | −3.3105 | −3.3484 | −3.3414 |
| val proposal loss | 2.2753 | 2.2649 | 2.2769 | 2.2776 |
| train proposal loss | 2.2103 | 1.8291 | 1.7709 | 1.7778 |
| `val_wrecall` | 0.8880 | 0.8917 | 0.8911 | 0.8912 |
| retained mask probability (val) | 0.8146 | 0.8713 | 0.8729 | 0.8760 |
| min/epoch (`_runtime` / 128) | 0.819 | 0.802 | 0.857 | 0.847 |

**What it shows.**

1. **Rejected.** The arm is −0.0160 against the control mean, −0.0156 against `c4a_base_s1`
   (paired), and −0.0136 against `no_bbe`: about three times the gate. It is below both
   controls at all 128 epochs (closest −0.0017) and above `no_bbe` at one. The screen did not
   cut it off early: its 10-epoch windows from 88 to 127 read 0.7113, 0.6977, 0.7132 and
   0.7110. Speed does not rescue it either. It is 4% faster per epoch than the controls and
   trails them at every epoch.
2. **The loss is in intensity fit, on both splits.** Primary loss is worse on train (+0.117
   against the control mean) and on val (+0.075), both far outside §3.2's 0.011–0.014 floor.
   So the arm fits the intensities worse; it does not just generalize worse. Its train/val
   primary-loss gap narrows to 0.580, against 0.614–0.623, while cosine falls. That is the
   fourth time in this file that a narrower gap has not paid. `val_wrecall` drops 0.003,
   outside the ±0.0006 span of §1's eight runs but small.
3. **The likely cause is encoder depth, but the two keys are not separated.** In both arms the
   proposal reads the encoder `.detach()`ed (see the
   [head-input asymmetry](masster_head_input_asymmetry_2026_09_22.md)). The encoder is
   therefore trained by the intensity loss alone either way, and removing the bond head cannot
   change what trains it. The bond head reaches the spectrum only through which fragments are
   selected. That selection did change: the selected masks hold 0.815 of the proposal's mask
   mass, against 0.873. But coverage moved by only 0.003 `wrecall`. So most of the −0.0136
   against `no_bbe` is probably the 2 → 5 depth acting on the intensity head. This is n=1, and
   the isolating arms are one key each on `c4a_base`: `masster_bond_num_layers: 0` or
   `mol_num_layers: 5`.
4. **The refinement head generalizes better early, then memorizes.** Val proposal loss:

   | epoch | `g1` | `c4a_no_bbe` | `c4a_base_s1` | `c4a_base_s2` |
   | --- | ---: | ---: | ---: | ---: |
   | 16 | 2.574 | 2.245 | 2.231 | 2.262 |
   | 32 | 2.331 | 2.230 | 2.233 | 2.209 |
   | best | 2.253 @ 85 | 2.205 @ 49 | 2.222 @ 35 | 2.207 @ 35 |
   | tail 118–127 | 2.275 | 2.265 | 2.277 | 2.278 |

   The three-block head reaches a lower val proposal loss, sooner, and the endpoint head never
   gets there. The two meet at the tail only because the refinement runs' val proposal loss
   rises after epochs 35–49 while their train proposal loss keeps falling to 1.77–1.83
   (`g1`: 2.21). Equal tail val loss does not mean the 3.40M parameters buy only train fit. It
   matches the late proposal degradation the
   [proposal early-stop arms](../../plans/masster/masster_proposal_early_stop_plan.md)
   measured.
5. **The one earlier one-key test did not favor the bond head (added 2026-09-25).** On
   2026-08-16, `vb6vbxpg` (`nist20_merged_mask256_gine_efa384_bond2`,
   `masster_bond_num_layers: 2`) was run against the endpoint head `18jmtpde` (key unset).
   Same seed; the other differing keys are exploration and curriculum options at their off
   values. Val cosine over epochs 30–37 was 0.6006 against 0.6047 for the endpoint head, and the
   bond head was ahead at 12 of 38 matched epochs. The endpoint run crashed at epoch 37, so there
   is no tail. This is the pre-2026-09-03 target and a much older model (val about 0.60), so it
   does not transfer to `c4a` directly. It is the only one-key evidence for the head, though, and
   the bond head was adopted without an epoch-matched win.

## 6. Historical encoder-size and head-size record (from local memory, verified 2026-10-06)

Source: the 2026-09-18 head-capacity audit in agent memory (every run reconstructed from its W&B config
through `MassterPL(**cfg)`; `s3gy1ro4` reproduced the 8,434,087 parameters in the
[merged-control guide](../../guides/benchmark_masster_merged_control.md)). W&B figures were not
re-queried. It predates the current 9.22M lineage, whose parameter budget is in the
[attention budget](../../plans/masster/masster_attention_budget_2026_09_21.md), so treat it as
background for any width or depth arm.

- **Encoder plateau, every cell ever run** (best `val_cos_sim_0.01`; mixed cohorts, so only the
  ordering inside one cohort is informative):

  | encoder | hidden | layers | n | best |
  | --- | ---: | ---: | ---: | ---: |
  | GINE | 256 | 4 | 7 | 0.6093 |
  | GINE | 384 | 5 | 3 | 0.6066 |
  | GINE_EFA | 384 | 1 | 1 | 0.6054 |
  | GINE_EFA/GSA | 384 | 2 | 185 | 0.8010 |
  | GINE_EFA | 384 | 3 | 1 | 0.6442 |
  | GINE_EFA | 384 | 5 | 2 | 0.6237 |
  | GINE_GRAPHORMER | 384 | 2 | 2 | 0.7297 |
  | Graphormer | 384 | 2 | 3 | 0.6922 |
  | Graphormer | 512 | 5 | 7 | 0.6989 |

  Encoder parameters anti-correlate with cosine in both older cohorts. Graphormer 384x2 is the clean
  contrast (same 2.83M bond head, twice the encoder, -0.05); every Graphormer 512x5 also sets
  `masster_bond_num_layers: 0`, collapsing the bond head from 2.83M to 595k, so those are confounded.
  Depth 1 to 2 is +0.016 and 2 to 5 is +0.003. Width has never been varied above 384 and GINE at
  hidden 512/768 is the untested cell. The depth result for the current lineage is `g1` in section 5.
- **Size against score on merged is a crossover.** On `tolerance_consensus` / `nist20_inchikey`, the
  7.4M `hi4mfsbj` beat the 22.7M `8v0i0fq7` at epoch 20 (0.7546 vs 0.7522), lost by epoch 40 (0.7682 vs
  0.7792), and 22.7M ended at 0.8010. But `hi4mfsbj` was a one-hour job (63 epochs at 52 s/epoch against
  252) still rising when killed; a small head at equal wall clock has never been run. Beyond 22.7M it
  reverses: the 24.5M class (22.7M plus a second 1.81M `proposal_encoder`) is worse at every matched
  epoch (epoch 40: 0.7762 vs 0.7792). Spearman(params, best) = +0.455 on the older cohort is a
  model-generation artifact (3.1M models had no intensity transformer) and falls to +0.250, p=0.059,
  in the comparable cohort.
- **Atom-decoder history.** Older merged target, 16.61M (decoder off, n=4) vs 24.77M (4 layers, n=44),
  best cosine by matched epoch: epoch 20 +0.0130, 40 +0.0100, 60 +0.0063, 100 +0.0008, best -0.0067;
  clock-matched the decoder-on model loses outright (113 vs 189 s/epoch). The decoder is now removed
  (see the [regularization audit](masster_regularization_audit_2026_09_19.md) and the
  [campaign read-out](masster_campaign_readout_2026_09_21.md)).
- **Unmerged family contrast.** `GINE_EFA` and `GINE_GSA` are one encoder renamed in `4b759e83`
  (2026-09-13) with the global pass changed from ELU+1 linear attention to exact softmax, so no run
  after that date can be EFA and none before it GSA; any EFA-vs-GSA delta absorbs two weeks of code
  (see the caveat in the [MASSter plan](../../plans/masster/masster_plan.md)). The unmerged A-vs-B
  gain (+0.0085) stays a bundle of attention kernel, dual CE, `drop_empty_spectra` and code. The
  intensity head (256x2, FF512, 64 inducing, `probability`, `cross_entropy`) was never varied across
  the 31 unmerged runs of 2026-09-18, while the merged 384x3/FF1024/ind128 + `independent` + JSS stack
  was worth +0.0129 epoch-matched on merged.
- **Config-resolution gotcha.** Reconstructing a current config needs a deep merge with the template
  (a shallow `{**base, **over}` drops `mol_params.pyg_pe_embed_k`), and old configs need
  `mol_gnn_type: GINE_EFA` changed to `GINE_GSA` plus `_legacy_candidate_backend=True`.

## Reproduction

`scan_history` over the run IDs in the tables above plus `5qws8g9x` / `6plumx6r`, keys
`val_cos_sim_0.01_epoch/mean`, `val_primary_loss_epoch/mean`, `train_primary_loss_epoch/mean`,
`val_wrecall_epoch/mean`, forward-filled by `epoch`. §5 adds `xb9ic2pp` and `pkavjg8z` and the
keys `val_proposal_loss_epoch/mean`, `train_proposal_loss_epoch/mean`,
`val_retained_mask_probability_epoch/mean` and `_runtime`; its parameter counts come from
building `MassterPL` from each resolved config. No script was added to the repository;
the query is three calls against `wandb.Api()` and is cheaper to rewrite than to maintain.

## Sources

- [Architecture review todo](../../plans/masster/masster_architecture_review_todo.md) — §1 is what this closes; §1's capacity table is the source of the parameter counts.
- [Overfit read-out](masster_overfit_readout_2026_09_22.md) — the run index, the replicate floor, the primary-loss method in §3, and the coverage conclusion in §4.1.
- [Campaign read-out](masster_campaign_readout_2026_09_21.md) — the 128-epoch screen length and the alternating-step logging trap.
