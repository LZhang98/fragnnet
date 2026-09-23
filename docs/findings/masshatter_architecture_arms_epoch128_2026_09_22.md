# MASShatter architecture arms, read at the matched 128-epoch screen, 2026-09-22

> **Status:** FINDING · **Updated:** 2026-09-22
> Three reads at the same screen against the same control. **§3 (added last) reads
> `a2_full_attention` and `b1_bond_sum`, which resolves §1's capacity confound — `a6_ce_film`'s
> +0.0063 is half topology and half CE-FiLM, and the pairwise mass bias is an exact null at
> matched topology — and §3.1 turns the accidental duplicate submission into the family's first
> measured run-to-run reproducibility floor: 0.0004 on the tail mean, 0.0025 on a single epoch,
> and 0.011–0.014 on the loss-drift columns §1 reads mechanistically.** **§2 (added later the same day) reads
> the five candidate-token feature arms and the intensity-escape pair, which the plans still
> recorded as "nothing trained" — they had in fact launched, and all six are nulls.**
> §1 closes §1 of the [architecture review todo](../plans/masshatter_architecture_review_todo.md):
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
[campaign read-out](masshatter_campaign_readout_2026_09_21.md) and the
[overfit read-out](masshatter_overfit_readout_2026_09_22.md).

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

§3 of the [overfit read-out](masshatter_overfit_readout_2026_09_22.md) measures each run's
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
[cut-set plan](../plans/masshatter_cut_set_features_plan.md) banner said "five matched arms built,
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

**No arm clears the floor in either direction.** `e6_ring_topology_cut_bond` was never submitted.

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

`a2` is the topology-only control §1 asked for — `attention_type: full` alone, the same 40.8%
smaller intensity head, nothing else. Against it, the three bias arms that share its topology:

| contrast | Δ tail | reading |
|---|---|---|
| `a2` − control | **+0.0030** | **`attention_type: full` is worth this much on its own, at 41% fewer intensity-head parameters** |
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

## Reproduction

`scan_history` over the run IDs in the tables above plus `5qws8g9x` / `6plumx6r`, keys
`val_cos_sim_0.01_epoch/mean`, `val_primary_loss_epoch/mean`, `train_primary_loss_epoch/mean`,
`val_wrecall_epoch/mean`, forward-filled by `epoch`. No script was added to the repository;
the query is three calls against `wandb.Api()` and is cheaper to rewrite than to maintain.

## Sources

- [Architecture review todo](../plans/masshatter_architecture_review_todo.md) — §1 is what this closes; §1's capacity table is the source of the parameter counts.
- [Overfit read-out](masshatter_overfit_readout_2026_09_22.md) — the run index, the replicate floor, the primary-loss method in §3, and the coverage conclusion in §4.1.
- [Campaign read-out](masshatter_campaign_readout_2026_09_21.md) — the 128-epoch screen length and the alternating-step logging trap.
