# MASShatter architecture review — remaining todo

> **Status:** ACTIVE · **Updated:** 2026-09-22
>
> Tracks only what is still open from the 2026-09-19 code-vs-literature review of the MASShatter
> stack. Five of its seven items were implemented on 2026-09-20/21, all default-off. **§1 is now
> closed**: the six arms that ran in the 2026-09-21 campaign were read at the matched 128-epoch
> screen on 2026-09-22 — `a1_prenorm` **passes cleanly** (+0.0056), `a6_ce_film` passes but carries
> a 40.8% intensity-head cut, `b2_identity_residual` is a **rejection** (−0.0062) and
> `b3_wd_groups` an exact null. Full numbers and method in the
> [epoch-128 read-out](../../findings/masshatter_architecture_arms_epoch128_2026_09_22.md).
> Items 4, 5 and 6 stay open and the five unrun arms were put on the 128-epoch screen on
> 2026-09-22. This doc does not cover the escape, candidate-token, loss-objective or
> regularization lines.

## Scope

The review named seven architecture-level gaps. Their current state:

| # | Review item | Implemented | Arm run | Verdict (matched epoch 127, 2026-09-22) |
| --- | --- | --- | --- | --- |
| 1 | Candidate head is mass-blind | partly (scalars + pair bias; composition off) | `a3` | **hold** — +0.0028, and confounded by the head cut |
| 2 | Post-LN under constant unwarmed LR | yes (`pre`, QK-norm; scheduler fixed) | `a1`, `a6`, `a7` | **PASS on `a1`** — +0.0056 clean, calibration drift 2.6x lower |
| 3 | GNN residual branch is ReLU-gated | yes (`gnn_residual_activation`) | `b2` | **REJECT** — −0.0062. Keep the ReLU gate. |
| 4 | RWSE off in every config | no — blocked on a one-line port | none | open |
| 5 | Bond head aggregates with a mean | yes (`bond_endpoint_aggregation`) | none | open |
| 6 | Induced-vs-full at a 256 budget | yes (`attention_type: full`) | none as a control | open, and it **confounds `a3`/`a6`/`a7`** |
| 7 | Weight decay on norms, biases, latents | yes (`weight_decay_exclude_norm_bias`) | `b3` | **null** — −0.0002; moves the loss, not the metric |

Two reference points for everything below. The screen length is **128 epochs**, confirmed on 15
arms (median +0.0031 from epoch 128 to best). The replicate floor for this family is
**0.003–0.005**, and the two `c4a_base` replicates span 0.0007 at epoch 127, which is tighter than
the family's history — a margin measured against them alone is flattered.

---

## 1. Read the six arms that already ran, at matched epoch 128 — **DONE 2026-09-22**

Full method, robustness checks and the primary-loss table are in the
[epoch-128 read-out](../../findings/masshatter_architecture_arms_epoch128_2026_09_22.md). Numbers
below are the tail mean of epochs 118–127 against a control mean of 0.7270 (replicates 0.7266 /
0.7273, spread 0.0007). Gate was ±0.005.

| arm | run | Δ control | verdict |
| --- | --- | --- | --- |
| `a6_ce_film` | `0gbdejd8` | **+0.0063** | PASS, but carries the −40.8% head cut below |
| `a1_prenorm` | `q4t4udee` | **+0.0056** | **PASS, clean** — parameter-identical, one key |
| `a7_ce_both` | `zgm04inn` | +0.0035 | hold |
| `a3_full_massbias` | `4fg0x5vw` | +0.0028 | hold |
| `b3_wd_groups` | `gjeqjp44` | −0.0002 | null — hold, not rejected |
| `b2_identity_residual` | `icii1zho` | **−0.0062** | **REJECT** — keep the ReLU gate |

Three things the read established beyond the gate:

- **The 630-vs-128 horizon does not confound the comparison.** All eight runs have
  `lr_schedule: False` at a constant `lr: 1e-3`, so nothing in the optimizer reads `max_epochs`;
  the horizon decided only when the job stopped. The `vulcan_l40s` tag on the six arms is also
  wrong — they ran on Fir H100, same hardware as the controls, within 35 minutes of them — and all
  six share `c4a_base_s1`'s seed 397623, so each is a seed-matched paired comparison.
- **`a1_prenorm` delays the calibration turn ~60 epochs and cuts post-turn primary-loss drift
  2.6x** (+0.0121 from its own optimum against a +0.0318 control mean), which is the mechanism the
  review predicted for post-LN under a constant unwarmed LR. `b3_wd_groups` shows the same shape at
  half the size with no cosine movement.
- **Coverage moves in none of them**: `val_wrecall` spans 0.0006 across all eight runs while cosine
  spans 0.0125. Every effect here is allocation.

**Effect on the rest of this doc:** `a4`/`a5`'s condition in §2 is met (`a1` cleared the floor,
`a3` did not). `a2_full_attention` was more load-bearing than before, not less — it decided
whether the campaign's best arm is a CE result or a topology result. **It ran on 2026-09-22 and
the answer is: both, half each.** See "§2's lead pair has now run" below.

### Half the ran arms carry an unstated capacity confound

Measured 2026-09-22 by constructing every arm from its resolved config. `attention_type: full`
deletes the induced layer's latent attention block, its feed-forward and the inducing tokens, so
**every arm that sets it is also a large capacity cut to the module the campaign identifies as the
allocator**:

| Arm | Total params | vs base | Intensity head | vs base |
| --- | --- | --- | --- | --- |
| `c4a_base_s1` | 9,215,573 | — | 2,457,355 | — |
| `a1_prenorm` | 9,215,573 | +0 | 2,457,355 | +0.0% |
| `b2_identity_residual` | 9,215,573 | +0 | 2,457,355 | +0.0% |
| `b3_wd_groups` | 9,215,573 | +0 | 2,457,355 | +0.0% |
| `a3_full_massbias` | 8,145,026 | −1,070,547 | 1,386,808 | **−43.6%** |
| `a6_ce_film` | 8,212,565 | −1,003,008 | 1,454,347 | **−40.8%** |
| `a7_ce_both` | 8,213,105 | −1,002,468 | 1,454,887 | **−40.8%** |

So `a1`, `b2` and `b3` are clean one-key arms and read directly against the control. `a3`, `a6`
and `a7` are **not** "does the mass/CE bias help" — they are "mass/CE bias, plus full attention,
plus a 41–44% smaller intensity head". That cuts both ways: `a6` ties `a1` for the best number in
the 25-run set while running 40.8% less allocator, which is itself informative, but none of the
three can be attributed until `a2_full_attention` supplies the topology-only control. This is the
same class of defect the 2026-09-13 candidate-attention audit caught once already.

**Consequence for §2:** `a2` is not optional and is not third in line. Promote it with `b1`.

### §2's lead pair has now run (2026-09-22)

Both finished at 127 epochs against the same 0.7270 control; numbers and method in
[epoch-128 read-out §3](../../findings/masshatter_architecture_arms_epoch128_2026_09_22.md).

- **`b1_bond_sum` is a null: −0.0013.** The degree-normalized mean on the bond head costs nothing
  measurable. Item closed; do not re-run it.
- **`a2_full_attention` is +0.0030**, and as the topology-only control it splits the two confounded
  leaders: `a6` − `a2` = **+0.0033** (CE-FiLM's own contribution at matched topology and capacity),
  `a3` − `a2` = **−0.0002** (the pairwise mass bias is an exact null), `a7` − `a2` = +0.0005.
- So the capacity confound above resolves in CE-FiLM's favour and against the mass bias:
  **`a6_ce_film`'s +0.0063 is about +0.0030 topology and +0.0033 CE-FiLM**, neither half clearing
  the ±0.005 gate alone, while `a3`'s entire +0.0028 was the attention change. **Review item 1's
  mass-blindness remedy is not answered by `intensity_mass_pair_bias`.** CE conditioning is the
  live half; `a5_ce_pair_bias` is what remains worth running, and `a4` drops in priority.
- `a1_prenorm` (+0.0056, one key, parameter-identical) stays the only clean single-key PASS and is
  now the cheaper of the two leaders.

## 2. Submit the five prepared arms that have never run

None of the five appears in any run index. At ~1 min/epoch a 128-epoch screen is about 2.2 h on the
box the campaign used. The jobs are generated by
`scripts/manual_slurm/fir/submit_masshatter_attention.sh [lead|conditional|rebalance|all]`, which
defaults to the `b1` + `a2` lead pair and refuses any config that has drifted off `max_epochs: 128`.

**Horizon — FIXED 2026-09-22.** Every config in `attention_2026_09_21/` was written before the
128-epoch screen was adopted and carried `min_epochs: 378 / max_epochs: 630`. Submitted as-is
against a 3 h box they are killed mid-schedule around epoch 170, which is exactly what happened to
`a1`, `a3`, `a6`, `a7`, `b2` and `b3` (killed at 179–210) and why §1 exists. The five unrun arms are
now `128/128`. The six that already ran were **left at 630 on purpose**, so their config still
matches the run in the index; set the horizon at submission time if any of them is replicated.

With that fixed, each unrun arm resolves to exactly its own keys against `c4a_base_s1` and nothing
else — verified by resolving both through `load_config` and diffing: `a2` and `b1` differ by one
key, `a5` by three, `a4` by four, `b4` by three. `h_transfer_bound`, `boundary_bond_encoder`,
the `c_mass` and `c_charge` candidate features are in the template's default
`mass_shatter_candidate_features` list now and resolve identically whether or not an arm
restates them. All five construct at 128/128.

| Arm | Key | Review item | Why it is worth the box |
| --- | --- | --- | --- |
| `b1_bond_sum` | `bond_endpoint_aggregation: sum` | 5 | The bond head is 41.9% of the 9.22M parameter budget and aggregates incident bonds with a degree-normalized mean, which is strictly less expressive than the sum the surrounding GINEConv was chosen for. Nothing else in the stack tests this. |
| `a2_full_attention` | `attention_type: full` | 6 | **Co-lead with `b1`.** The only control for the −41 to −44% intensity-head cut that `a3`, `a6` and `a7` all carry (§1). Without it, three of the six ran arms — including one of the two campaign leaders — cannot be attributed at all. Note it is 8,144,981 params against the base 9,215,573, so it tests full attention *at reduced capacity*, not at matched capacity. |
| `a4_full_massbias_prenorm_qknorm` | four keys | 1, 2 | The stacked arm, and it also carries the capacity cut. **Its §1 condition is met** — `a1` cleared the floor, `a3` did not — but §1 also found the effects are *not* additive (`a7`, which stacks pre-LN onto CE-FiLM and both biases, scores below `a1` and `a6` alone), so expect a stacked arm to underperform its parts here. |
| `a5_ce_pair_bias` | `intensity_ce_pair_bias` | 1 | Completes the CE triple with `a6`/`a7`. §1 raised its value: at matched capacity and matched topology `a6` (CE-FiLM) beats `a3` (mass pair bias) by **+0.0035**, so CE conditioning is the live half of review item 1 and `a5` separates FiLM from a pair bias as the route. |
| `b4_budget_rebalance` | `bond_num_layers: 1`, `transformer_dim: 384`, `num_inducing_points: 128` | — | Not a review item, and **not a one-key arm** — it moves capacity from the bond head to the intensity head and widens the latent set at the same time, so a result cannot be attributed. Listed only because it shares the box and the base. Lowest priority; split it before running it. |

**`a5` prepared for submission on its own (2026-09-22).** `submit_masshatter_attention.sh` now
has an **`a5` stage** that generates `a5_ce_pair_bias` alone, because the `conditional` stage also
generates `a4` and the epoch-128 read argued against `a4` in the same motion that raised `a5`:
stacking has now cost twice in this family (`a7` below both `a1` and `a6`; `e5` below `e4`).
`a5` resolves to **three** keys against `c4a_base_s1` — `attention_type: full`,
`intensity_mass_pair_bias`, `intensity_ce_pair_bias` — so **read it against `a3_full_massbias`
(`4fg0x5vw`), which holds the first two without the CE term.** `a5` − `a3` isolates the CE pair
bias; `a5` − control does not, and would re-credit it with the +0.0030 that belongs to the
topology change.

Order: **`b1` and `a2` first.** `b1` is parameter-identical to the base, so it is the cleanest
one-key arm in the whole set; `a2` unblocks the reading of three arms that already ran — and after
§1 that matters more, because `a6` is the campaign's best arm and `a2` is what decides whether that
result belongs to CE-FiLM or to full attention. `a5` next (§1 made CE the live half of item 1),
then `a4`.

**Also worth a box after §1:** a seed replicate of `a1_prenorm` at 128 epochs. It is the first
clean, parameter-identical, one-key architecture arm in this family to clear the floor, it is n=1
against an n=2 control whose 0.0007 spread is tighter than the family's history, and it is the
natural partner for the `d3` LR-decay result — the two live candidates both act on optimization
under a constant LR and neither has been run with the other.

**Note on `a3`/`a5`/`a6`/`a7`:** `PairwiseMassBias` weights initialize to zero, so these are exact
no-ops at step 0 and safe on a warm start. The diagnostic for the CE arms is
`intensity_head.mass_pair_bias.ce_weights` read out of the checkpoint — a flat `w_r(CE)` is
informative about [CE-invariance](../../findings/masshatter_overfit_readout_2026_09_22.md) whether or
not cosine moves, so these arms pay out in both directions.

## 3. Port the RWPE shape fix, then screen RWSE

**The only review item with nothing implemented.** Two steps, in order.

1. **Port the fix.** `random_walk_pe` at [feat_utils.py:1003](../../../src/fragnnet/utils/feat_utils.py)
   still builds its adjacency with `scipy.sparse.csr_matrix((data, (row, col)))` and lets scipy
   infer the shape from the largest index. The one-argument fix (`shape=(g.num_nodes, g.num_nodes)`)
   exists on `feiw_v2` and `project/spectraverse_benchmark` and is **not** on this branch.
   `pyg_bigraph: True` saves connected molecules, but a zero-bond molecule raises. Port it with its
   tests before setting the flag anywhere.
2. **Screen it.** `pyg_pe_embed_k` is `0` in `config/template.yml` and in all MASShatter configs.
   MASShatter builds mol graphs live in the dataset, so this is **config-only — no preprocessing and
   no cache regeneration**. Prepare one arm at `k: 16`, matching the FraGNNet `*new_mol_feats*`
   configs.

**Mechanism, which is MASShatter-specific:** the bond-break head must separate 1-WL-equivalent
bonds — para-substituted rings, alkyl chains — that fragment differently. This is a proposal-side
change, so read `val_candidate_wrecall` alongside cosine, not cosine alone.

**Caveat before betting on it:** the encoder is a measured plateau over width, depth and
architecture ([head capacity](../../findings/masshatter_campaign_readout_2026_09_21.md)). RWSE is
untried because it changes *input features*, not capacity, which is the one axis that plateau never
covered — but that is an argument for it being unmeasured, not for it working.

## 4. Feed composition into the candidate token

Item 1 is only partly closed. The `c_mass` and `c_charge` candidate features landed
2026-09-20 and are in the default `mass_shatter_candidate_features` list, so absolute mass now
reaches the head. Still missing: any **composition** vector and the **neutral-loss complement**
formula.

- The `c_nl_composition` entry (GLACIER eq. 14's `psi(r_form - f_k)`) is implemented and absent
  from the default list; `c4a_complement_composition` is prepared and has never run.
- Submit it as a 128-epoch arm alongside §2.

**Two smaller items in the same token, neither prepared:**

- `cut_counts` is fed raw into the input projection at
  [bond_fragment_heads.py:501](../../../src/fragnnet/model/bond_fragment_heads.py) and `:977` while
  component sizes get `log1p` on the adjacent line. Make it consistent, or state why not.
- Absolute mass arrives as two raw scalars. A sinusoidal or RBF encoding is the standard treatment
  and is what the pair bias already does for *relative* mass. Untested either way.

## 5. Decide the mask-residual shortcut, or drop it from this list

`mass_shatter_use_mask_residuals: True` in production lets the intensity transformer override the
bond model's base probability, which [§6.1.1 of the MASShatter plan](masshatter_plan.md) argues can
improve allocation while starving candidate selection of supervision.

This is the weakest item in the review and may already be moot: production also sets
`proposal_selection_only: true` and `proposal_anchored: true`, and the
[head-input asymmetry finding](../../findings/masshatter_head_input_asymmetry_2026_09_22.md) shows the
proposal branch is detached in every current `ma_mi_merged` arm. Reference configs exist only on
the older unmerged/merged bases.

**Action:** read those three interactions against the current base and either write a matched arm
on `c4a_base_s1` or close this item with the reason. Do not port the old reference configs.

---

## What this doc deliberately excludes

- The scheduler forwarding repair — **fixed in `031c370d`, verified 2026-09-21**. `lr_schedule_type`
  and `lr_total_steps` now reach `build_lr_scheduler`. Do not list it as an action again.
- Mixed precision. The CUDA candidate backend requires `precision: 32-true`, and bf16 has known
  `index_add` and normalization failures on this model.
- The LR-decay family (`d1`–`d3`), regularization arms, loss objectives, the intensity escape and
  the candidate-token cut-set work. Those are tracked in their own plans and in the
  [overfit read-out's](../../findings/masshatter_overfit_readout_2026_09_22.md) ordered interventions.

## Sources

- [Attention rework + parameter budget](masshatter_attention_budget_2026_09_21.md) — what landed for items 1, 2, 3, 5 and 7, and the arm definitions.
- [4 h exploration campaign](masshatter_campaign_4h_2026_09_21.md) — the base config, the box, and why 128 epochs.
- [Overfit read-out](../../findings/masshatter_overfit_readout_2026_09_22.md) — the run index these arms sit in, and the replicate floor.
- [Epoch-128 read-out](../../findings/masshatter_architecture_arms_epoch128_2026_09_22.md) — **closes §1**: the six ran arms against both controls at matched epoch, with the horizon/hardware/seed confound checks.
