# Current project state

> **Status:** REFERENCE · **Updated:** 2026-09-23

## Current handoff

**Proposal early-stop arms read; both nulls, and the family is closed (2026-09-23).** Read-only
W&B query, nothing changed but the docs. Tail mean 118-127 against the 0.7270 control: `p1`
freeze@32 `s5i5qqlz` **+0.0010**, `p3` proposal wd 0.1 `w1dj2ahw` **+0.0020**. `p2` is not worth
submitting. The useful by-product is that `p1` measured the shared encoder drifting under
frozen proposal heads. Train proposal loss rises in every block after epoch 32 (+0.082 in
total), and about a third of the control's late val-proposal degradation survives the freeze.
That means a clean freeze needs `separate_proposal_encoder`. It does **not** mean the shared
encoder costs cosine. Proposal loss and cosine are decoupled across the four runs, and the one
same-seed, one-key separate-vs-shared test (`esb2fqhg`/`d7s0yb72`, 2026-08-21, old merged
lineage, never recorded before) is **−0.0008**. A separate-encoder arm is a one-key 128-epoch
test with a null prior, ranked behind the `a1_prenorm` and `d3` replicates.
[Plan § Results](plans/masshatter/masshatter_proposal_early_stop_plan.md#results-p1-and-p3-at-the-128-epoch-screen-2026-09-23).

**SMARTS usage audited end to end (2026-09-23); read-only, nothing changed but one new probe.**
No production model uses a SMARTS rule. In MASShatter, the eleven compound ring-cut rules add
only a merged neutral byproduct (chance-floor intensity); their use is forcing a 4-bond mask.
The neutral-loss rules are the real new chemistry, bounded at about 0.1% of NIST20 intensity.
Past FraGNNet prepass verdicts measured `crf12`/`crf13`, which make up 99% of its partitions.
Also: a dead sulfoxide rule, `uniquify` dropping anthraquinone partitions, and stale
`smpFalse`/`ringmerge` stats files. Next is the model-free coverage check in §5.1 before any
CUDA port.
[Finding](findings/smarts_usage_audit_2026_09_23.md).

**MASShatter fragment-marginal selection implemented (2026-09-23); nothing trained or submitted.**
A U128 cap loses cosine because it truncates the pool in mask order, and the proposal loss gives
zero gradient to peaks only the cut-off fragments explain. The new opt-in
`mass_shatter_fragment_selection: marginal` keeps the forced precursor/single cuts plus the top
`mass_shatter_selected_fragment_target` fragments by closed-form release probability, and scores
the proposal loss on the whole enumerated pool. Default `mask_order` is unchanged; the full suite
passes (4,271, after two review fixes). Three arms against `c4a_base` are prepared: `f1` mask-order U128, `f2` marginal
K128 from U256, `f3` from U512. Generator: `scripts/manual_slurm/fir/submit_masshatter_fragment_selection.sh`.
Reading rules and open points:
[plan](plans/masshatter/masshatter_fragment_marginal_selection_plan.md).

**FraGNNet FT contender-pair Stage A prepared for Fir (2026-09-23).** Configs and a generation-only
script; **nothing submitted.** Four arms from `fraggnn_d3_ma_mi_nist20_amolc_all.yml` (the
`jnokddr9` config): `c0` original sampler, `c1` `ce_context` sampler control, `p1`/`p2`
`ce_pair_loss_weight` 1.5 / 6.0. The weights come from a measured initial gradient ratio
g = 0.0344 on ~406-spectrum steps (5% / 20% targets). Two things change the matrix from the plan's
reading. First, `target_rel_ints_thresh` (required by the pair loss) also filters the
**validation** target, so all four arms carry 0.01 and `jnokddr9` is not the control. Second, the
original group sampler shows only 138,875 of 287,721 train spectra per epoch, so the context arms
run 48 epochs to match c0's 100 on spectra seen. The primary read is p1/p2 vs c1. Details, open
gate items and costs:
[plan § Stage A preparation](plans/fragnnet/ft_contender_pair_minimal_training_plan.md#stage-a-preparation-2026-09-23).
Generator: `scripts/manual_slurm/fir/submit_fraggnn_ce_pair.sh` (run on Fir; needs the DAG tarball
and `wheels/` there).

**`a5_ce_pair_bias` and the full-length `wd10` arm prepared for Fir (2026-09-22).** Config and
script only; **nothing submitted, no GPU used, no run launched.** Two arms, three jobs.

**`a5`** needed no config change (already 128/128). `submit_masshatter_attention.sh` gained an
**`a5` stage** so it can be generated without `a4`, which the epoch-128 read argued against in the
same motion that raised `a5`. `a5` resolves to **three** keys against `c4a_base_s1`, so it must be
**read against `a3_full_massbias` (`4fg0x5vw`)**, not the control — `a3` holds
`attention_type: full` and the mass pair bias without the CE term, and `a5` − `a3` is the only
contrast that isolates `intensity_ce_pair_bias`. `fir:h100-3g.40gb`, 3 h.

**`wd10` cost more than it looked and the reason is memory, not wall clock.** The question is the
train/validation gap, which needs a **train** cosine, and `compute_train_auxiliary_metrics` is
**false** in `c4a_base_s1/s2` — so the existing control supplies a matched val cosine and no
matched train cosine at all. `ctrl_a`/`ctrl_b` do have it on but predate the 2026-09-20 default
flip, so they are a different architecture. Two new/edited configs, both never run:
`..._c4d_wd10.yml` moved 630 → **128/128** with the metric on, and a new
`..._c4a_base_s1_trainaux.yml`. Resolved diffs verified: control − base = that key alone,
**arm − control = `weight_decay` alone**. Same seed 397623, so the control's val cosine is also a
third same-seed replicate of `c4a_base_s1`. Generator:
`scripts/manual_slurm/fir/submit_masshatter_regularization.sh wd10`.

**Both wd10 jobs are `fir:h100-80gb`, not the 3g.40gb slice, and that doubles their price (12.2
vs 6.1 each).** Measured from the W&B system stream on 2026-09-22: the metric takes peak GPU from
**30.4 GiB** (`c4a_base_s1`, off) to **36.2–37.2 GiB** (`ri3zm215`, `kalc4mzo`, on), against a
3g.40gb slice's **37.25 GiB** — under 40 MiB of headroom before the CUDA context. Slice size does
not affect the metric, only throughput, and every read here is at matched epoch. The same
measurement clears the wall-clock worry: the key costs **+6%** per epoch (0.91 vs 0.86 min/epoch),
so 128 epochs is ~1.95 h in a 3 h box. **The 2026-09-19 arms died at epoch 64-65 because they had
a one-hour box, not because of this key** — that screen was ~10% exposure.

Two cautions carried into the read. `kalc4mzo` is **not** this arm's predecessor: it ran
2026-09-19, before the three candidate/h-transfer defaults flipped, so re-resolving its config
today inherits defaults it never trained with. And the prior does not favour the arm — **three
arms have now narrowed the gap and lost** (`drop03`, `b3_wd_groups`, `e2_cc_atom_drop01`); it is
run to close the question, not because it is expected to win. Also noted while copying:
`c4a_base_s1.yml` carries a `train_aux_metrics` W&B **tag** while the key is `false`, the wrong
tag the overfit read-out already caught; the copy's tag was corrected, the original left alone so
it still matches its indexed run.

Validation: both new configs parse and resolve; `max_epochs: 128` and the train-metric key
asserted by the generator; `bash -n` clean on both scripts; resolved one-key diffs verified
through `load_config`. Job generation itself must run on Fir — the wheel preflight needs
`wheels/`, which is Git-ignored and absent from this checkout.

**`a2_full_attention` and `b1_bond_sum` read; §1's capacity confound resolved, and the family's
reproducibility floor measured (2026-09-22).** Read-only W&B query, no GPU, nothing changed.
Both lead-pair arms finished at 127 epochs on the same screen (tail mean 118-127, control
**0.7270**, gate ±0.005), so the question the previous handoff left open is answered.
**`b1_bond_sum` is a null at −0.0013** — the degree-normalized mean is not costing anything, do
not re-run it. `a2_full_attention` lands at **+0.0030**, and as the topology-only control it
splits `a6_ce_film`'s +0.0063 about evenly: **+0.0030 from `attention_type: full` itself
(at 41% fewer intensity-head parameters) and +0.0033 from CE-FiLM**. Neither half clears the gate
alone. The sharper result is `a3`: **the pairwise mass bias is an exact null (−0.0002) at matched
topology**, so all of `a3`'s earlier +0.0028 was the attention change, and the review's
mass-blindness item is *not* answered by that term. `a1_prenorm` (+0.0056, one key, no capacity
change) is still the only clean single-key PASS and is now also the cheaper of the two leaders.

The accidental duplicate finally pays for itself. `xyessyn1` and `cgomm8v4` are the same config
and the **same seed** (`deterministic: False` in both — the `deterministic` W&B *tag* on these runs
is wrong), and they diverge from epoch 0 and never reconverge. **The floor depends entirely on the
statistic:** 0.0025 mean |Δ| on a single epoch (0.0053 at epoch 64), 0.00036 on the tail mean of
118-127, 0.00001 on best ≤127. That **validates the 118-127 tail mean every verdict in this family
uses** — its run-to-run floor is tighter than the 0.0007 cross-seed control spread — and rules out
single-epoch reads, which carry more noise than five of the eleven arm effects on record. It also
puts a much coarser floor under the *mechanistic* columns: **0.011 on the loss drift and 0.014 on
the train/val loss gap**. The turn *epoch* is reproducible (78 vs 78 at fixed seed, 52 vs 66 across
seeds), so `a1_prenorm`'s turn at epoch 120 stands; but the "2.6x drift cut" is ~1.7 floor-widths
and should be read as directional, and the drift ordering among `a3`/`a6`/`b2`/`a7` is inside the
floor. Full tables in [epoch-128 read-out §3](findings/masshatter/masshatter_architecture_arms_epoch128_2026_09_22.md).

**Next.** `a5_ce_pair_bias` and `a4` are what remain of review item 1, and a seed replicate of
`a1_prenorm` is now clearly worth a box — it and `d3` are the two live candidates and both are n=1.
Promote nothing on a single-epoch read; the gate on tail-mean comparisons could be tightened well
below ±0.005, but not on one duplicate pair.

**Proposal warm-up read at the matched screen and rejected as the alternative (2026-09-22).**
Asked whether to freeze the proposal branch or warm it up and lean on weight decay. The three
`c4c_pwarm` arms had never been read at the 128-epoch screen — they ran 630-epoch configs and
were killed near epoch 205, so their summaries were last-logged values at unmatched epochs.
Tail mean of epochs 118-127 against the same 0.7270 control (gate ±0.005): `e2` **+0.0008**,
`e4` **+0.0012**, `e15` **−0.0096** — two nulls and a rejection, non-monotone, and the horizon
does not confound it (all carry `lr_schedule: false` at constant `lr: 1e-3`, all seed 397623).
On top of that the key un-detaches the proposal branch for the whole run whenever the encoder is
shared, which is every `ma_mi_merged` config, and that rewiring measured −0.0113@100 / −0.0132@400
on its own. Warm-up also treats the start of the run where the defect is at epochs 26-47. Read-only
W&B query, no GPU, nothing changed by that read. Recommendation recorded in the plan: run `p1`
(freeze@32) and `p3` (proposal wd 0.1), hold `p2`, drop warm-up.

Proposal early stop and proposal-only weight decay implemented (2026-09-22, later the same day).
**Code and config; no run launched, no GPU used.** The
[overfit read-out §3](findings/masshatter/masshatter_overfit_readout_2026_09_22.md) records that the
validation **proposal** term degrades +0.06 to +0.14 and **turns at epochs 26-47** while the
spectrum term keeps improving to 52-120 — two branches on one training schedule. Under
`proposal_selection_only: true` the proposal reads detached inputs, so its loss updates the
proposal heads and nothing else, and those heads are **44.8% of the parameters** (4,128,642 of
9,217,181 at `c4a_base_s1`). Two template keys now let them stop or decay on their own, both
default `null`: `mass_shatter_proposal_frozen_from_epoch` freezes the proposal-only modules from a
chosen epoch, and `mass_shatter_proposal_weight_decay` gives them their own decay through a new
`_weight_decay_overrides` hook that leaves `SpectrumPL.configure_optimizers` single-group when
nothing opts in. Neither touches `precursor_encoder` or `context_encoder`; a forward/backward test
asserts the proposal objective updates **nothing** past the boundary while the spectrum objective
still reaches the intensity head and both encoders. Three one-key 128-epoch arms are written and
verified in `config/massshatter/ma_mi_merged/proposal_early_stop_2026_09_22/` (`p1` freeze@32,
`p2` freeze@64, `p3` proposal wd 0.1) and **none has been submitted**. Two limitations belong in
the read and are in the [plan](plans/masshatter/masshatter_proposal_early_stop_plan.md): the shared encoders
keep training, so the proposal *function* still drifts (a static proposal needs
`separate_proposal_encoder`, off everywhere), and freezing also puts the heads in eval mode, which
drops their `head_dropout: 0.1` — a second, unseparated change that `p3` does not carry. The same
read-out predicts these may be nulls, since coverage is already flat train-to-val; read
`val_wrecall` next to cosine. Validation: full `pytest tests -q` clean, `ruff check` clean on the
four changed Python files, Black deliberately not run (those files were already not Black-clean at
`HEAD`). No run launched.

Feature and escape arms read (2026-09-22, later the same day). **The plans' "nothing trained"
and "No production run launched" lines were stale — the arms had been submitted, and six of them
now have verdicts. All six are nulls.** Read-only W&B query, no GPU, nothing changed in the tree.

**The five candidate-token feature arms finished at 127 epochs**, on the same screen and against
the same control as the architecture arms (tail mean of epochs 118-127, control **0.7270**,
replicate spread 0.0007, gate ±0.005): `e4_cut_adjacency` **+0.0032**, `e5_cut_bond_plus_adjacency`
+0.0021, `e3_cc_atom_drop02` +0.0002, `e2_cc_atom_drop01` −0.0009, `e1_cut_bond_feats` −0.0010.
**Not one clears the floor.** Adjacency is the only arm with a sign, which is the direction the
[descriptor measurement](findings/masshatter/masshatter_candidate_cut_descriptors_2026_09_22.md) predicted
(0.489 residual ambiguity against 0.000 for ring-sharing), so the mechanism argument survives and
the effect does not. Two patterns repeat from the architecture read: **stacking costs** — `e5` =
`e1` + `e4` scores 0.0011 *below* `e4` alone, as `a7` scored below `a1` and `a6` — and **narrowing
the loss gap is still not the objective**: `e2` has the narrowest train/val primary-loss gap of the
seven runs (0.5719 vs a 0.6191 control mean) and the lowest cosine of the five, the third instance
after `b3_wd_groups` and `drop03`. Coverage is flat again (`val_wrecall` spread 0.0006).
`e6_ring_topology_cut_bond` was never submitted.

**The intensity escape is an exact null at n=2 vs n=2.** `e1_escape` against `e0_independent`, two
seeds each on the `batch256x2_lb32` base, read at their common epoch 118 while still running:
0.7225 against 0.7225, **Δ +0.0000**, seed spreads 0.0001 and 0.0003. Re-read at completion, but
nothing in the trajectory suggests a late separation.

**Two caveats worth carrying forward.** (1) Each e-arm differs from `c4a_base_s1` by **eight**
resolved keys, not one — seven new keys resolving to `False`/`0` plus three inert `spec_params`
additions from the above-precursor filter. All are no-ops at those values so the comparison holds,
but check whether `spec_params` participates in any dataset cache identity before the next such
arm: new inert keys in a hashed dict are how a "one-key" arm quietly changes its inputs.
(2) **`a2_full_attention` was submitted twice at the same seed 397623** (`xyessyn1` on `fc11018`,
`cgomm8v4` on `fc10906`) — a duplicate, not a replicate, costing a box that `a5_ce_pair_bias` or an
`a1_prenorm` replicate could have used. It does accidentally measure the run-to-run reproducibility
floor, which is worth having given how much weight the 0.0007 control spread carries.

`b1_bond_sum` (`3bqnobq1`) and `a2_full_attention` launched at 128/128 epochs and were at epoch 20
when this was written — too early to read. Numbers, method and confound checks in
[epoch-128 read-out §2](findings/masshatter/masshatter_architecture_arms_epoch128_2026_09_22.md). Banners
corrected on the [cut-set plan](plans/masshatter/masshatter_cut_set_features_plan.md) (now SCREENED, ALL NULL)
and the [escape plan](plans/masshatter/masshatter_supervised_escape_plan.md) (now IMPLEMENTED, SCREENED NULL).
Validation: relative links resolve, Ruff not applicable, no test run and none required for a
documentation change. No run launched.

Architecture arms read at the matched 128-epoch screen (2026-09-22). **Documentation only; a
read-only W&B query, no GPU, no code, config or run changed.** This closes §1 of the
[architecture review todo](plans/masshatter/masshatter_architecture_review_todo.md), which was the blocking
item in the previous handoff: the six arms killed at the wall clock between epochs 179 and 210 now
have verdicts against both `c4a_base` replicates at matched epoch. Numbers are the tail mean of
epochs 118-127 against a control mean of **0.7270** (replicates 0.7266 / 0.7273, spread 0.0007),
gate ±0.005 — full method and robustness checks in the
[epoch-128 read-out](findings/masshatter/masshatter_architecture_arms_epoch128_2026_09_22.md).

**`a1_prenorm` +0.0056 PASSES and it is the clean one** — parameter-identical to the base, one key,
and the first such arm in this family to clear the floor. It is also the strongest mechanistic
result in the set: it moves the validation primary-loss optimum from epoch 52/66 to **120** and
cuts the post-turn drift **2.6x** (+0.0121 from its own optimum against a +0.0318 control mean),
which is exactly what the review predicted for post-LN under a constant unwarmed LR.
**`b2_identity_residual` is a rejection at −0.0062** — keep the ReLU gate on the GNN residual
branch, do not re-run it. **`b3_wd_groups` is an exact cosine null (−0.0002)** that still narrows
both the drift and the train/val loss gap, so it is a hold, not a rejection.
`a6_ce_film` posts the best number (**+0.0063**) but carries the **−40.8% intensity-head cut** that
`attention_type: full` imposes, so it stayed unattributable until `a2_full_attention` ran —
which it since has; see the current handoff and §3 of the read-out. `a7_ce_both` (+0.0035) and `a3_full_massbias` (+0.0028) are
holds.

**Three things the read establishes beyond the gate.** (1) The 630-vs-128 horizon does **not**
confound the comparison: all eight runs have `lr_schedule: False` at a constant `lr: 1e-3`, so
nothing in the optimizer reads `max_epochs` and the horizon decided only when the job stopped. Two
further metadata traps are harmless but worth knowing — the `vulcan_l40s` tag on the six arms is
**wrong** (`run.metadata['gpu']` says all eight ran on Fir H100, within 35 minutes of each other),
and all six arms carry `c4a_base_s1`'s seed 397623, so each is a seed-matched paired comparison.
(2) The effects are **not additive**: `a7`, which stacks pre-LN onto CE-FiLM and both bias terms,
scores below `a1` and `a6` individually; and among the three arms that share the identical attention
change, CE-FiLM beats the mass pair bias by **+0.0035** at matched capacity and matched topology,
which makes CE the live half of review item 1. (3) **Coverage moves in none of them** —
`val_wrecall` spans 0.0006 across all eight runs while cosine spans 0.0125 — an eight-run
confirmation of the overfit read-out's allocation-not-coverage conclusion on a population it did
not cover.

**Next.** `b1_bond_sum` and `a2_full_attention` remain the lead pair, `a5_ce_pair_bias` gains
priority over `a4`, and a seed replicate of `a1_prenorm` is now worth a box: it is n=1 against an
n=2 control whose 0.0007 spread is tighter than this family's history, the same caveat the `d3`
LR-decay result carries, and `a1` and `d3` are the two live candidates that both act on
optimization under a constant LR and have never been run together. Validation: relative links
resolve, Ruff not applicable, no test run and none required for a documentation change. No run
launched.

Architecture-review todo consolidated (2026-09-22). **Documentation only; no code, config or
run changed.** The 2026-09-19 code-vs-literature review's remaining items are now one list:
[architecture review todo](plans/masshatter/masshatter_architecture_review_todo.md). Five of its seven items
were implemented 2026-09-20/21 and all default off. The blocking observation is that the six arms
that did run (`a1`, `a3`, `a6`, `a7`, `b2`, `b3`) were killed at the wall clock between epochs 179
and 210 and **have never been read at the 128-epoch screen length**, so four review items rest on a
W&B read that costs no GPU -- `a1` and `a6` lead the 25-run set at 0.7364 but over a longer horizon
than the 0.7282 control they would be promoted over. **That read was done later the same day; see
the handoff entry above for its verdicts.** Verified while writing it: every config in
`config/massshatter/ma_mi_merged/attention_2026_09_21/` still carries `min_epochs: 378 /
max_epochs: 630`, and resolved against the current template that schedule is their **only**
difference from `c4a_base_s1` besides their own keys, because `h_transfer_bound`,
`boundary_bond_encoder`, `candidate_mass_features` and `candidate_charge_feature` are template
defaults now; that schedule is why the ran arms were killed mid-run, and the five unrun arms
(`b1_bond_sum`, `a2_full_attention`, `a4`, `a5`, `b4`) need it set to 128 before submission.
`b4_budget_rebalance` is **not** a one-key arm -- it moves `bond_num_layers` 3 -> 1 while widening
the intensity head and the latent set.

**Fixed and measured 2026-09-22, same request.** The five unrun arms are now `min_epochs`/
`max_epochs` 128; the six that already ran were left at 630 so their config still matches the run
in the index. Each unrun arm now resolves to exactly its own keys against `c4a_base_s1` (`a2`,
`b1` one key; `a5` three; `a4` four; `b4` three) and all five construct. Constructing every arm
then surfaced an **unstated capacity confound**: `attention_type: full` deletes the induced layer's
latent attention block, feed-forward and inducing tokens, so every arm that sets it also cuts the
intensity head -- the module the campaign identifies as the allocator -- by **43.6%** (`a3`,
8,145,026 total) or **40.8%** (`a6`/`a7`). `a1`, `b1`, `b2` and `b3` are parameter-identical to the
9,215,573-param base and read directly; `a3`, `a6` and `a7` are *not* a mass/CE-bias test but
"bias + full attention + a 41-44% smaller head", so `a6` ties `a1` for the campaign's best number
(0.7364) while running 40.8% less allocator. None of the three is attributable until
`a2_full_attention` supplies the topology-only control, which promotes `a2` to co-lead with `b1`.
Same class of defect as the 2026-09-13 candidate-attention audit's capacity finding. RWSE is the
only item with nothing implemented and stays
blocked on porting the `random_walk_pe` `shape=` fix, which is on `feiw_v2` and
`project/spectraverse_benchmark` but **not on this branch**. The plan excludes the scheduler
forwarding repair, which is fixed in `031c370d`. Validation: relative links resolve, Ruff not
applicable, no test run and none required for a documentation change. No run launched.

MASShatter candidate-token changes (2026-09-22). **LANDED in the working tree, both keys default
off, nothing trained.** Two one-key levers on the head that the
[overfit read-out](findings/masshatter/masshatter_overfit_readout_2026_09_22.md) blames for ~98% of the
0.151 cosine gap (allocation, not coverage: train `wrecall` 0.8975 vs val 0.8911).
**`mass_shatter_candidate_cut_bond_features`** pools `chemical_bond_attr[boundary_bond_indices]`
per candidate, summed and averaged, because `_boundary_embeddings` builds its block from atom
ENDPOINTS and never reads the bond attributes -- so a cut bond's own type/ring/conjugation reach
the intensity head only through endpoint mixing, which an internal atom of degree > 1 destroys.
The summed half is the multiset of broken bond types. Costs 2 x bond width (18 columns here,
1,285 -> 1,303) and no parameters. **`mass_shatter_candidate_atom_dropout`** drops each atom from
a candidate's membership set before the component pool, training only, attacking the identity that
pooling leaves behind (the cc atom mask resolves 1.00 distinct fragments per descriptor bucket).
Dropped atoms **migrate to the complement** rather than the complement being held exact, because
the complement is precursor minus candidate and an exact one would leak the true candidate sum by
subtraction; every other candidate input keeps the true membership, and a candidate is never
emptied. Arms, each verified a one-key diff from `c4a_base_s1` at the same seed and 128 epochs:
`config/massshatter/ma_mi_merged/candidate_token_2026_09_22/{e1_cut_bond_feats,e2_cc_atom_drop01,e3_cc_atom_drop02}.yml`.
**Correction to the [cut-set plan](plans/masshatter/masshatter_cut_set_features_plan.md):** its item C
(remove `use_base_probability_feature`) is not available as an arm -- `mass_shatter_pl.py:1008`
derives it from `mass_shatter_proposal_anchored`, which is already `true` in `c4a_base_s1`, so the
`c4a` token never carried `mask_log_probabilities`.

**Then the premise was remeasured on the right population, and it moved the design.**
[Candidate cut descriptors](findings/masshatter/masshatter_candidate_cut_descriptors_2026_09_22.md),
`scripts/debug/debug_candidate_cut_descriptors.py`, 400 molecules / 73,045 of MASShatter's OWN
candidates rather than the depth-3 DAG the plan measured: **86.9%** are cut at >= 2 bonds (the DAG
said 96.8%) and the relational shares are about twice the DAG's -- **73.6%** of all candidates have
two cuts sharing a ring, **68.4%** two cuts sharing an atom. Within-molecule candidates per
descriptor bucket: cut count **47.5** -> pooled cut-bond sum **22.6** -> plus adjacency **10.4** ->
plus both ring terms **10.4**. Given the bond sum, the ring-sharing flag has **0.000** residual
ambiguity (the sum carries `b_ring` counts) while the adjacency flag has **0.489**. So the plan's
item A was redirected: adjacency is the missing relation, and it needs no SSSR ring identity --
which is not on the graph and would cost a new field plus a cache regeneration for 0.02-0.03 of
bucket resolution. Shipped as **`mass_shatter_candidate_cut_adjacency_features`**: two columns,
cut-bond pairs meeting at a shared atom, split by whether that atom is retained (one atom lost two
substituents) or lost (the fragment expelled a bridging atom -- the CO/CH2/S extrusion shape).
Arms `e4_cut_adjacency` and the stacked `e5_cut_bond_plus_adjacency` join e1-e3. **Counter-weight,
recorded in the finding:** the 2026-07-23 bond-set probe found the *least* identifiable fragment
descriptor generalized best, so resolving more candidates locates information the head cannot
otherwise get -- it does not predict the sign on validation cosine.

Validation: 19 new tests, the candidate-feature file 54 pass, **full suite 4,206 passed / 1
skipped / 2 xfailed**, Ruff and Black clean on every changed file, all five arms verified as
one-key (e5: two-key) diffs after template resolution. Smoke:
`config/debug/debug_tiny_mass_shatter_cut_features.yml` stacks all three keys at toy width with
dropout 0.3 and ran **green on `debug_tinyv4`**, 2 epochs, `detect_anomaly: true`, CUDA candidate
backend, no NaN, loss falling (train -0.509 -> -0.754, val -0.806 -> -0.981). Decomposed one key
at a time against the all-off control at three seeds: **every arm lands inside the control's own
seed band** (val cosine 0.196-0.225), the cut-bond block gives the best train loss of the seven
runs and adjacency the best val loss, and the only systematic effect is the dropout suppressing
the training fit. Added columns do not make the toy fit harder; the risk they carry is
generalization, not optimization. Parameter deltas land exactly on the token and are additive (+288 cut-bond, +32
adjacency, +320 both, +0 dropout at test width). No production run launched.

Feature and information audit (2026-09-22). **Documentation and diagnostics only; two of its
own conclusions withdrawn.** Synthesis with a per-claim status table:
[feature_information_audit_2026_09_22.md](findings/feature_information_audit_2026_09_22.md) --
read it before any of its five parts. Headline results: GLACIER publishes **no model size at all**
(ICEBERG 2.0 is 7.0M against FraGNNet 2.79M and MASShatter 9.22M, all measured); descriptor
features are a **closed null** (MMFF/EState/Crippen/TPSA and the benzylic family are recoverable
from current inputs at 0.93-1.00); `b_stereo` cannot be matched without a preprocessing change
(0 of 127,249 bonds carry stereo); NIST ships an expert formula per peak covering **99.4% of
intensity** that never reaches the loss. In the tree, untrained: three new atom/bond features with
tests, `mass_shatter_candidate_composition_features` (off by default), three matched config arms,
and five diagnostic scripts. Proposed, not built: three MASShatter candidate-token changes
([cut-set plan](plans/masshatter/masshatter_cut_set_features_plan.md)) -- **96.8%** of fragments are cut at
>=2 bonds and **55.4%** have two cuts sharing a ring, which mean pooling cannot distinguish from
unrelated cuts. **Retracted:** the proposal that MASShatter needs a structural teacher; coverage
already transfers (train `wrecall` 0.8975 vs val 0.8911 against a 0.151 cosine gap). Validation:
full suite green, Ruff clean, links checked. No run launched.

Above-precursor target filter (2026-09-22). **LANDED in the working tree.** `spec_params` gains
`filter_above_precursor_peaks` / `above_precursor_ppm` /
`above_precursor_use_instrument_tolerance`, default off, applied in `clean_spectra` after
centroiding and before subformula filtering, with the same per-instrument tolerance and the same
retention statistics as its two sibling cleanups, plus the standalone CLI flags. The cutoff is
`prec_mz` widened by one matching tolerance -- deliberately the same rule
`scripts/eval/dag_oracle_scores.py` uses for its `1_above_prec` tier, which was the default for
every published oracle number while the training target kept those peaks. **Turning it on also
removes the precursor isotopologue ladder**, which is 72% of the affected intensity and is real
analyte signal; that is intended while isotopes are unmodelled (`num_isotopes` is 1 everywhere,
every DAG `isoFalse`), but it changes the target, so **cosines are not comparable across this
flag**. Validation: full suite 4,167 passed / 1 skipped / 2 xfailed; Ruff clean on every changed
file; 9 new tests; end-to-end on the real merged MA/MI val split the above-precursor intensity
goes **0.00899 -> 0.00000** and 10,339 of 792,217 peaks are removed, matching the 0.0090 predicted
by the split probe. Three existing `spec_params` fixtures needed the new required keys, which is
the intended strict-required behaviour. No run launched.

Intensity escape / dump (2026-09-22), same request. **LANDED END TO END** behind
`mass_shatter_intensity_escape`, default off, **never trained**.
`sparse_peak_list_distance` takes `pred_dump_intensities` (shape `(batch_size,)`, same units as
`pred_intensities`); when given it collapses every *unreachable* observed peak into one appended
coordinate and lets the model predict that mass, with all four objectives and the entropy term
running unchanged on the extended vectors. Motivation is the user's: `wrecall` is 0.891, so 10.9%
of observed intensity sits where no candidate reaches and the predicted distribution -- normalized
over candidates alone -- is forced to over-allocate everything it can reach. Worked exactly for
Hellinger, the escape removes an irreducible per-spectrum loss floor of `1 - sqrt(t(S))` that
varies with coverage and therefore weights training by how well enumeration did on that molecule;
**it does not change the optimal relative allocation**, which is `q_i ∝ t_i` either way, so it is
not a ranking change. It also makes `kl_distance` finite on a coverage gap rather than leaning on
`kl_epsilon`. Reachability is read from the tolerance `match_mask`, never from predicted values,
so a candidate the model scored zero -- or one a Hungarian assignment gave to a neighbour -- still
faces the loss. **This dissolved the plan's open "where is the target computed" decision**: it is
batch-time machinery that already exists, with no `spec_df` column and no `tolerance_consensus`
merge interaction. Validation: full suite **4,177 passed / 1 skipped / 2 xfailed**, Ruff clean,
10 new tests covering the algebra, the structural reachability rule, gradient flow, KL finiteness
and input validation; `pred_dump_intensities=None` is bit-identical to the previous behaviour on
all four objectives, under test. The model side is wired: `MassShatterModel` takes
`intensity_escape` and drives the intensity head's existing `predict_oos`, hard-coded `False`
until now; the head emits one logit per spectrum from its mean-pooled molecule token and the
model returns `pred_escape_intensities`. Parameterization: the escape's share of `candidates + escape`
is `sigmoid(logit)`, whose odds are just `exp(logit)`, so the emitted mass is
`candidate_mass * exp(logit)` with the logit clamped at 9. A code review caught the first
implementation going through `sigmoid` and back, which **broke under reduced precision**: fp16
and bf16 round any `1 - 1e-4` guard to exactly `1.0`, so the share saturated and the odds became
`inf`, crashing the step on the loss's finiteness check. The head's bias starts at `logit(0.1)`,
so training begins with the escape holding 10% of the spectrum, close to the measured 0.109, and
a test asserts it. The escape head also warm-starts from a checkpoint that predates it, matching
what `ion_h_support_head` already did, so switching it on does not force a run from scratch. `MassShatterPL` **raises at
setup** when the escape is requested with `mass_shatter_intensity_output: probability` or a
cross-entropy `loss_type` rather than ignoring it -- silent override is how `predict_oos` went
unnoticed on this model for months. The escape is reachable only in unique-fragment,
proposal-selection-only mode, the only mode `independent` intensity output supports. Also fixed: the
head's molecule pooling divided by a component count that can be zero, on a path that had never
executed because `predict_oos` was hard-coded off at its only call site. Validation: full suite
**4,184 passed / 1 skipped / 2 xfailed**, Ruff clean, 17 new tests in total for this item,
including an end-to-end one that runs the real model, feeds its output through `loss_fn`, and
shows the escape both reaches `sparse_peak_list_distance` and lowers the loss. **Smoke-trained green 2026-09-22**, not measured:
`config/debug/debug_tiny_mass_shatter_intensity_escape.yml` and a matched control differing only
in the flag, 64 toy spectra, 8 epochs, `detect_anomaly: true`, both exit 0. The head starts at
0.1 by construction and reaches `escape_share` **0.591** against a true unreachable fraction
`1 - wrecall` of **0.528**, so it tracks the coverage deficit; `wrecall`/`recall`/`precision` are
identical to the last digit between arms, the control the design requires; Hellinger halves
(0.746 -> 0.362) against a predicted floor of `1 - sqrt(0.4717)` = 0.313. **The loss drop is not
a quality gain** -- the objective now includes a coordinate the model can trivially get right, so
a lower number is guaranteed, and **cosine, the one metric whose definition did not change, went
slightly DOWN (0.2236 -> 0.2022)**. That is noise at this scale but is what the algebra predicts:
the escape is not a ranking change. `escape_share` is logged per spectrum whenever the escape is
on, since otherwise the head is invisible. **Real comparison must be on cosine at matched epoch
with replicates, never on the loss.** **Merged arms prepared, not submitted:**
`config/massshatter/ma_mi_merged/intensity_escape_2026_09_22/` is a **one-knob ladder** off the
loss-objective arm `h1_hellinger_closest.yml` (`oykkffls`, 0.7247), needed because `h1` keeps
`intensity_output: probability` while the escape requires `independent` -- so `e0` (independent,
escape off, 2 seeds) isolates the output mode against `h1` and `e1` (escape on, 2 seeds) isolates
the escape against `e0`. Verified by resolving each config and constructing the model: the
resolved diff is exactly the intended key at each step, and the escape adds **257 parameters** to
a 9,217,181-parameter model with no change to the memory shape, so the lineage's
`fir:h100-2g.20gb` profile still applies. Generation:
`scripts/manual_slurm/fir/submit_masshatter_intensity_escape.sh [ladder|control]` (generates
without submitting, validates configs and wheels first, syntax-checked). Nothing generated into
`slurm_scripts/`, nothing submitted. The
contaminant escape (`predict_oos` supervised by the subformula test, addressable population
0.0156) is still unimplemented; see [the plan](plans/masshatter/masshatter_supervised_escape_plan.md).
Isotope detection is **deferred by user decision** (2026-09-22).

Unexplained-intensity split (2026-09-22). Answering "what escape do we need for the peaks we
cannot explain" by measuring the population first; read
[the split](findings/masshatter/unexplained_peak_split_2026_09_22.md). Over the full 15,257-spectrum
`nist20_ma_mi_inchikey` val split, built through the run's own `spec_params` so the probed target
is the trained target, the 10.89-point residual behind `val_wrecall` 0.8911 divides **73%
coverage / 27% escape**: 2.91 points carry no precursor-bounded, nonnegative-RDBE subformula and
therefore cannot come from the molecule, while **7.98 points are real analyte fragments the
candidate enumeration misses**. The test is not vacuous -- a matched-mass donor formula rejects
0.5993 where the true formula rejects 0.0291, a **20.6x** discrimination. The escape population
is a diffuse small-peak background (an infeasible peak carries 0.21x a feasible one's intensity,
half the mass is spread over 7.9% of spectra, 23.3% of spectra are entirely clean) with a ~1.6%
tail where a dominant peak is non-analyte, **so a per-spectrum scalar escape is the wrong shape
for it** -- which agrees independently with the read-out's proof that a scalar `predict_oos`
cannot move a scale-invariant metric. Flat across adduct, instrument and fragmentation mode.
**46.5% of the escape population is isotopologues, not contaminants** (0.0135: the precursor M+1..3
ladder plus fragment isotopologues one or two neutron masses above an explainable peak), leaving a
genuine noise population of **0.0156** -- 14.3% of the residual. That half wants isotope handling,
which is scoped but never switched on (`remove_isotope_peaks` and friends are marked NOT HOOKED UP
in `template.yml`; `num_isotopes` is 1 everywhere and every DAG is `isoFalse`), so the model is
graded on peaks it structurally cannot produce. **Nothing removes peaks above the precursor**
anywhere in the training or validation path, while every oracle script (`dag_oracle_scores.py`,
`dag_oracle_decomposition.py`, `dominant_peak_diagnostic.py`) defaults to the `1_above_prec` tier
that does -- so published oracle ceilings were measured on a target the model does not train on.
A blanket above-precursor cut is nonetheless wrong: it removes 0.0090 of which 72% is the precursor
isotopologue ladder, real signal; only 0.0025 is genuine co-isolation. **Priority is coverage, not
escape** -- 7.98 points versus a 1.56-point contaminant population. Relevant existing machinery,
all currently off for MASShatter: the `predict_oos` head is implemented in
`bond_fragment_heads.py:678` but hard-coded off at `mass_shatter_model.py:1508` and force-disabled
in `MassShatterPL`; `filter_subformula_peaks` is a hard per-peak version of this probe's test and
has never run on this base (0 of 58 runs). New probe:
`scripts/debug/debug_unexplained_peak_split.py`. Validation: Ruff clean; probe run end to end on
the full split plus a 400-spectrum smoke run; no model code changed, no training launched.

MASShatter overfit read-out (2026-09-22). Read-only W&B review of the 25 runs of 2026-09-21/22;
read [the read-out](findings/masshatter/masshatter_overfit_readout_2026_09_22.md) before launching another
regularization, objective or schedule arm. The train/validation cosine gap reaches **0.151** by
epoch 127 and never plateaus, but it is **not a coverage gap**: train and validation `wrecall`
agree to 0.0064 against a 0.1507 cosine difference, worth ~0.0027 of achievable cosine at the
local `wrecall`->`opt_cos_sim` slope measured on three 2026-08-26 runs, so **~98% of the
generalization gap is intensity allocation among peaks the model already proposes**. FraGNNet on
this exact split logged the oracle on both sides and agrees (train `opt_cos_sim` 0.9271 /
validation 0.9243). Whether narrowing the gap helps is **untested, not refuted**: the
four-objective ordering is algebraically forced (val-on-train slope 0.605 makes gap and
validation rank together by construction), and in the 8-arm 09-21 regularization sweep
corr(gap, val) = +0.72 collapses to **-0.67** once the single arm `drop03` is removed. `wd10`
narrowed the gap without suppressing train cosine and moved validation **+0.0015** at 10%
exposure -- one full-length run would settle it. Validation *primary* loss turns before
validation cosine in all six instrumented runs but by **6 to 476 epochs** (the CE control leads
by 6, i.e. not at all); the 630-epoch control ends **0.096 worse** than its epoch-79 optimum,
non-monotonically, while its cosine moves **+0.0042** and never falls -- a calibration signal,
not evidence that late training costs cosine. The 128-epoch screen length is confirmed on 15 arms
(median +0.0031 from epoch 128 to best). **One live candidate:** the LR-decay arms, the first to
get a real schedule after the 2026-09-21 forwarding fix -- `d3` (cosine decay + 1000-step warmup)
reaches 0.7357 against a 0.7286 control mean whose replicates span 0.0007, but n=1; replicate
before promoting. **Two instrumentation gaps block the next read:**
`compute_train_auxiliary_metrics` is on in 4 of 25 runs and `opt_cos_sim` in none, and
`c4a_base_s1/s2` carry a `train_aux_metrics` tag while their resolved config says `False`. Also
pinned: the 0.73-vs-0.80 difference against `8v0i0fq7` is **scope plus the subformula target
filter**, not a regression -- a 256-run cross-tab puts ~0.06 on adduct/instrument/fragmentation
scope (4 adducts / FT+QTOF / HCD+CID vs `[M+H]+` / FT) and ~0.006 on `filter_subformula_peaks`,
which is `True` in `8v0i0fq7` and `False` in **all 58** `nist20_ma_mi_inchikey` runs. The filter's
share is not a matched eval: `clean_spectra` runs at dataset load, so each arm is scored against
its own cleaned target. The read-out now also records the intensity-memorization threat model:
candidate-set fingerprinting, unequal molecule exposure, CE conditioning proxies, population
priors, and non-causal fragment proxies are plausible but not yet causally isolated. The current
lead already excludes direct proposal probabilities, and unmerged runs also overfit, so neither
removing that feature again nor simply unmerging is a supported fix. Next gate: eval-mode,
molecule-balanced train/validation metrics with seen/unseen-scaffold slices, followed by matched
`wd10`, instrumented `d3`, intensity-only capacity/dropout, molecule balancing, and QTOF CE arms.
Documentation only; no code, config or run changed, and no training or evaluation was launched.

ICEBERG/GLACIER parity (2026-09-22). Closing the gaps behind "do they have a larger model, more
features?". **Model size:** GLACIER publishes none — no hidden dim, layer count or parameter
count, only `4xNVIDIA A5000`; its own baseline ICEBERG 2.0 is 7.0M on NIST'20 (measured from the
upstream configs), against FraGNNet 2.79M and MASShatter 9.22M (both measured). **Features:**
ICEBERG carries strictly fewer than we do; GLACIER carried three we lacked, now added to the
shared registry — `a_fused_ring`, `b_ring_size`, `b_fused_ring` — plus the candidate token's
complement composition, `mass_shatter_candidate_composition_features` (GLACIER eq. 14's
`psi(r_form - f_k)`), off by default. `b_stereo` cannot be matched by a feature: preprocessing
strips stereochemistry, so 0 of 127,249 sampled NIST20 bonds carry any, on NIST23 and
MassSpecGym alike. Three matched arms are prepared and **none has been run**:
`c4a_ring_topology`, `c4a_complement_composition`, and FraGNNet `graph_feature_parity`.
Details in [the GLACIER note](notes/glacier_object_detection.md); validation: full suite green,
Ruff clean on every changed file; no evaluation, no cluster job. Two debug configs stack **all
four** new features at toy width and both smoke-trained green on 2026-09-22 (2 and 5 epochs on
`debug_tinyv4`, loss finite and falling, `detect_anomaly: true`, CUDA candidate backend):
`config/debug/debug_tiny_mass_shatter_glacier_feats.yml` and
`config/debug/debug_tiny_fraggnn_graph_feature_parity.yml`. Smoke runs, not measurements.

The shared agent workflow is established: start at [AGENTS.md](../AGENTS.md), follow the
[session workflow](workflow/session.md), and use the [architecture map](architecture.md).
Existing AGENTS and Claude rules are organized into [engineering](rules/engineering.md),
[Python and ML](rules/python.md), and [documentation](rules/documentation.md) rules.
The agent entry point selects the applicable files by task; the original rules are preserved.
This includes import aliases, formatting/docstrings, testing conventions, and documentation
placement. Where legacy Claude constraints differed, the original AGENTS policy is retained:
preprocessing changes require explicit authorization and compatibility work; PR validation runs
when practical, with omitted checks reported.
The adoption changes remain uncommitted for user review.

Validation: shared-rule preservation, relative documentation links, and Git whitespace checks
passed. The configured interpreter provides pytest and Ruff. No model code changed; no training,
scientific evaluation, or full test suite was run for this documentation change.

MASShatter CMF alignment (2026-09-14): MASShatter now emits the full configured H-transfer
ladder for chemically valid, non-degenerate CMF adducts, matching FraGNNet. It preserves
reachable CRF/CMF duplicate suppression and in-source-loss exclusion; the pre-batch support
filter shares the same contract. Existing checkpoints are diagnostic-only because their newly
valid CMF-H channels were not trained. Validation: 3,032 passed, 21 skipped, 2 xfailed; the
CMF plan records the required oracle-coverage rebaseline and matched-control training.

## Research navigation

- [Energy-based spectrum reranking](plans/energy_based_spectrum_reranking_plan.md) is a proposed
  shared design of record with a FraGNNet anchor (2026-09-13). Next: reconcile the prior correction
  null, verify export with `--min_intensity -1`, and require oracle excess over matched random pools
  before implementing/training an energy head. Documentation only; no runs or implementation.
- [MASShatter improvement plan](plans/masshatter/masshatter_improvement_plan.md) records the proposed
  unmerged sequence: corrected CE control, proposal-encoder comparison, Gaussian support-gated
  JSS, conservative output pruning, and measured throughput improvements. Five merged warmup
  configs are prepared for a CE/JSS/KL loss and intensity-transform screen. Next: verify checkpoint
  persistence before launching; resolve the unmerged anchor/warmup policy before its matched
  control. The plan includes [Vulcan generation and submission commands](plans/masshatter/masshatter_improvement_plan.md#vulcan-commands-for-the-five-config-screen)
  for all five configs with a three-hour allocation. No runs were launched for this plan.
- [MASShatter code audit](findings/masshatter/masshatter_code_audit.md) records three reproduced
  issues at `0b5ad3f4`: standalone SMARTS inference, charged-component masses, and
  SMARTS proposal gradient isolation. All three fixes and regression tests are present
  at `77fdbea2`. A continued audit reproduced two additional inference blockers:
  unconditional DAG-file filtering/dataset selection in the CLI and independent-output
  tabular export expecting `pred_logprobs`. Those fixes and the mass-feasibility
  alignment are now in the working tree, along with CLI directory/row-layout fixes.
  Validation: 781 passed, 8 skipped; six CLI regressions passed after adding Parquet
  serializer-boundary coverage. Real Parquet serialization was not verified because
  its optional engine is unavailable; checkpoint loading and dataset preprocessing
  are mocked in the CLI tests. Next check: a target-environment real-checkpoint CLI
  smoke run. Review the linked compatibility
  notes and re-evaluate affected checkpoints before comparing experiment metrics.
  A third audit pass found two remaining API issues: prediction-only
  `run_inference_pkl` rejects its existing `pred_ints` output, and direct CPU bfloat16
  calls with multiple proposal mixtures fail the normalization check. Standard CPU
  runner fallback avoids the latter. The pickle API fix is now in the working tree:
  90 tests passed, including real-model prediction-only regressions for both output
  modes and batch offsets. The bfloat16 issue is deferred by user because it is not
  used in the current workflow. Separate failure reproductions remain in the audit.
  The checkout was clean at audit start, so the
  uncommitted-work description in the workflow-adoption handoff above is historical.

- [MASShatter regularization sweep](plans/masshatter/masshatter_regularization_sweep_plan.md) is **EXECUTED with
  no promotions**; read the [campaign read-out](findings/masshatter/masshatter_campaign_readout_2026_09_21.md)
  (2026-09-21) before acting on the plan. It was the execution of that audit's sweep half:
  eight one-hour `fir:h100-3g.40gb` jobs on the
  merged MA/MI base `nist20_ma_mi_merged_fast_batch512_cuda_lb64`, comprising the six prepared
  one-knob arms (`layernorm`, `graphnorm`, `drop02`, `drop03`, `wd10`, `ctxnorm`) plus two controls,
  all with `compute_train_auxiliary_metrics: true` so a train cosine is logged for the first time in
  this model family. Four historical replicates of the base config set the floor at about 0.003 val
  cosine sd from epoch 20 on, so the screen resolves roughly 0.007 and promotes on mechanism plus
  margin; throughput moved 2.5x between two same-config runs on one day, so every comparison is at
  matched epoch, never at matched wall clock. The GPU profile is an open, measured question rather
  than a fixed choice: `loss_batch_size` is pure loss chunking (`loss.py:629-675`), so batch 512 fits
  a `2g.20gb` slice at `lb16` (~15.8 GiB projected against 26.9 GiB measured at `lb64`), but that
  slice carries 2/8 memory slices against the 3g profile's 4/8 for 57% of the price, on a workload
  whose NVML memory utilization runs 72-90%. Stage A therefore submits the control on both slices
  and lets the measured `min/epoch` ratio choose between 8 jobs at n=1 and 14 jobs at n=2. The plan
  excluded an LR arm because the `lr_schedule_type` forwarding was dropped in
  `SpectrumPL.configure_optimizers`; **that forwarding is fixed as of 2026-09-21**
  (`spectrum_pl.py:1782-1815`, `fragnnet_pl.py:1511`), so an LR arm is now available and the
  exclusion no longer applies. The plan records that the
  `cc_*` dropouts listed as unswept are FraGNNet-only and inert for MASShatter. Configs resolved and
  one job script generated and syntax-checked locally outside the repository; nothing was generated
  in `slurm_scripts/` and no job was submitted at plan time.
  **Outcome (2026-09-19/20).** Six of eight arms produced data: `_drop02` was killed by the
  out-of-range predicted-peak bug about 70 minutes before its fix landed in `ffe5d53a`, and
  `_ctxnorm` was never submitted. A1-A5 pass — train cosine logged for the first time, `E* = 64`,
  controls agree to 0.0016, backend resolved `cuda` on all ten jobs. **A6 fails**: the 2g pilot
  OOM'd in `calculate_match_mzs` and the fan-out went ahead on `3g.40gb` three minutes later, so
  the slice ratio is still unmeasured. Read-out: 0 promote, 5 hold, 1 reject (`drop03`, −0.0149 vs
  the control mean). Only `wd10` (gap 0.122) and `drop03` (0.113) move the train/val gap off the
  control's 0.128; `layernorm` and `graphnorm` are exactly null. Every arm was still climbing at
  the one-hour wall, at 10% of its 630-epoch schedule, and the best validation epoch is the last
  epoch for all eight.
  **Horizon probe read out 2026-09-21 (`tkkesus9`, full 630 epochs, 8.97 h, 70 ep/h): val cosine
  saturates by ~epoch 100, not 630.** 0.7232@64 -> 0.7292@100 -> 0.7287@300 -> 0.7309@629, so the
  last 529 epochs buy **+0.0017**, inside the 0.0016 control replicate floor; the nominal peak
  0.7318@555 is noise. The sweep's +0.0034/10ep extrapolation decays immediately -- total headroom
  past ep64 was only +0.008. **The ep64 nulls therefore stand, and the correct screen length is
  ~128 epochs (~1.8 h), not 630 (~9 h)**; the saved wall clock should buy n>=2 replicates, since
  at a 0.003-0.005 replicate floor an n=1 arm cannot resolve what this family produces. The
  train/val gap **never plateaus** (0.55@64 -> 0.61@100 -> 0.81@300 -> 0.88@600) and validation
  loss actively degrades after ~ep100 (-2.526 -> -2.424) while train loss keeps falling: pure
  memorization, hidden by the best-val-cosine checkpoint. Caveat: the probe carries the
  2026-09-20 feature bundle, so its curve cannot be overlaid on the §1 arms to retro-size that
  screen -- the saturation shape is the transferable result, not the absolute level.
  **Proposal warmup is null-to-harmful on `ma_mi`** (`jvaj1lyw` vs `tkkesus9`: the only differing
  resolved-config key is `mass_shatter_proposal_only_until_epoch: 4`, same seed). Best-vs-best
  -0.0004, final -0.0021, and behind at every usable horizon: -0.0113@100, -0.0130@300,
  -0.0062@500, catching up only in the last ~80 epochs. The +0.0071 that motivated it was on the
  merged `nist20_inchikey` split and does not transfer; the four prepared `c4c_pwarm_*` arms are
  answered and should not run.
  **Re-read 2026-09-22 — that key does two things, and the second one is the story.** With
  `separate_proposal_encoder: False` (true for both runs),
  `set_proposal_input_gradients_enabled(proposal_only or proposal_encoder is None)` is `True` in
  **every** epoch, so the key also un-detaches the proposal branch for the whole run. `jvaj1lyw`
  is therefore the only `ma_mi_merged` run in which proposal supervision reaches the shared
  encoder, and a 4-epoch freeze cannot explain a −0.0113 deficit at epoch 100 or −0.0132 at 400
  in a run that saturates by epoch 100. **The persistent component is the gradient connection**,
  so the detached proposal branch should be left alone. Corollary: the 38 configs with staging
  *and* a shared encoder were all silently gradient-connected and are not clean tests of their
  nominal variable. See
  [head-input asymmetry §7](findings/masshatter/masshatter_head_input_asymmetry_2026_09_22.md).
  Next: run the `c4a_*` bundle-attribution arms, then the now-unblocked LR-schedule arms
  (`c4b_*`), then refill the `_drop02` / `_ctxnorm` holes.
- [MASShatter regularization audit](findings/masshatter/masshatter_regularization_audit_2026_09_19.md)
  (2026-09-19) establishes that regularization and optimization have effectively never been varied:
  across 226 `model_type = mass_shatter` W&B runs, `mass_shatter_head_dropout`, `weight_decay`, `lr`,
  `gradient_clip_val` and `train_sample_weight` are identical in all of them, and only two runs vary
  any regularization or optimization setting without also changing the encoder. Both predate the
  `tolerance_consensus` merged-target change, so neither is usable evidence. Five prepared ablation
  configs under `config/massshatter/ma_mi_merged/` have never been launched. Measured mechanisms:
  BatchNorm, not dropout, drives the divergence between the candidate set optimized in training and
  the one evaluated (Jaccard 0.317/0.611 at dropout 0 under `batch`, exactly 1.000 under `layer`,
  `graph` and `none`); normalizing a residual branch pins its magnitude for `batch`, `layer` and
  `graph` alike, so switching normalization does not restore a layer's ability to attenuate itself;
  and `SpectrumPL.configure_optimizers` dropped `lr_schedule_type`/`lr_total_steps`/`lr_min_ratio`,
  giving 3.774e-04 instead of the configured 1.000e-05 at step 100,000. **That forwarding is fixed
  as of 2026-09-21**; runs before that date trained on exponential decay whatever their config said.
  Next: measure the BatchNorm candidate-set gap on a trained checkpoint before any new training arm.
  It also records the 2026-09-19 decision to drop the atom decoder going forward: epoch-matched it
  costs no cosine (-0.0018, inside the replicate floor) while running 1.8x faster per epoch, and
  `config/template.yml` already defaults it to 0, so the recommended normalization arm
  needs no config change. The decoder code path remains for checkpoint compatibility.
  Probes: `scripts/debug/probe_masshatter_regularization.py`. Audit only; no training run,
  no model-code change.
- [Cross-adduct loss headroom](findings/cross_adduct_loss_headroom_2026_09_17.md) is a NEW, GREEN
  data-side gate (2026-09-17) for grouping MS/MS by adduct and adding a paired loss. 46.5% of
  NIST20 molecules carry at least two adducts; 78,938 groups match on instrument, fragmentation
  mode and NCE; the InChIKey split never separates a molecule's adducts, so groups are
  leak-free and usable in validation. `[M+H]+` and `[M-H]-` of the same molecule share about
  40% of intensity as neutral fragments at 29x over control, but a molecule-independent
  neutral-loss transform fitted on disjoint molecules closes only 9.3% of the gap to the 0.950
  replicate floor, so roughly 92% of the cross-adduct difference is molecule-specific and an
  adduct embedding cannot supply it for free. Chosen formulation: a delta loss on the shared
  fragment subset, keeping adduct-conditioned proposals. Blockers to implementation are known:
  `group_id` never spans adducts (0 of 76,164 groups) and the shared-proposal path asserts one
  precursor type per group. Next: run the trained-model half of the gate against a multi-adduct
  checkpoint; both checkpoints in this checkout are `[M+H]+` only. No implementation started.
- [MASShatter feature set audit](reference/masshatter_feature_set_audit.md) (2026-09-20)
  audited every feature that reaches a MASShatter parameter, over all `model_type: mass_shatter`
  configs, with column constancy measured on a 5,000-molecule NIST20 sample (120,641 atoms).
  It found one real defect: `AtomDegree.featurize` passes a list into `one_hot_encoding`, so 12 of
  `a_degree`'s 14 columns were always the unknown bucket and substitution degree reached the GNN
  only as two unnormalized integers; in total 34 of 70 atom-feature columns were constant. It also
  found the candidate intensity head received no mass, m/z or composition at all; that `a_charge`
  was enabled in FraGNNet's newest configs and in no MASShatter config, although MASShatter is the
  model with an explicit charge-affinity head; and that the boundary embedding was degenerate in
  the two ways FraGNNet diagnosed for `boundary_pair` and fixed with `CutBondEncoder`.
  **All of these are now fixed for MASShatter and are the default** (audit §11): `a_degree`'s
  one-hot repaired **in place** at unchanged width (the interim `a_degree_onehot` featurizer was
  deleted; the repair is shared with FraGNNet, so every pre-2026-09-20 checkpoint of either model
  must be retrained), `mass_shatter_candidate_mass_features`,
  `mass_shatter_candidate_charge_feature`, `mass_shatter_boundary_bond_encoder`, `a_charge` added
  to all 177 MASShatter configs, the constant unique-mode token columns dropped, the unread
  required `atom_formal_charges` check removed, and the featurizer no longer mutating
  `CANONICAL_ELEMENT_ORDER`. Constant atom columns fall from 34/70 to 30/71. FraGNNet is
  deliberately untouched: `a_degree` still exists and no FraGNNet config changed, though the same
  defect applies to all 323 of them. **Every MASShatter checkpoint written before 2026-09-20 is
  incompatible** (atom width 70 -> 71 and a changed intensity-head input width), and molecular
  preprocessing caches must be regenerated for `atom_gasteiger_charges`. Validation: full suite
  3,890 passed, 1 skipped, 2 xfailed; 29 new tests; the lead merged config resolves and runs a
  finite forward pass. `a_radical` and `a_chiral` were deliberately left in place; the audit §11
  records why.
  **Measured 2026-09-21: the bundle is worth +0.0073 val cosine at matched epoch 64**, clearing
  both the 0.003 floor this entry asked for and the 0.0016 control replicate floor. The horizon
  probe `tkkesus9` (`..._lb64_long_control`, `nist20_ma_mi_inchikey`) is the first run carrying
  the new defaults: 0.7232@64 against the 2026-09-19 regularization-sweep control mean 0.7159
  (`ri3zm215` 0.7152 / `7cdsx970` 0.7167). A full resolved-config diff of `ri3zm215` -> `tkkesus9`
  leaves exactly six substantive keys, all from audit §11: `a_charge`,
  `mass_shatter_boundary_bond_encoder`, `mass_shatter_candidate_mass_features`,
  `mass_shatter_candidate_charge_feature`, `h_transfer_bound` none->valence (independently
  measured free at -0.0002, so ~no contribution), and the `a_degree` one-hot repair, which is
  **invisible in the config** -- same name and same 14-column width, but 12 columns were dead
  before `97c41abc`. (`daa36f63` "update features" is comment reflow only, no behaviour change.)
  **This is a bundle, not an attribution**, at n=1 against n=2 controls and riding ~2,400 lines of
  other `src/` churn between the two batches, so no single component is established. The
  `c4a_*` arms isolate it: full bundle + seed replicate, four leave-one-out arms
  (`no_mass_feats`, `no_charge_feat`, `no_bbe`, `no_acharge`) and an all-off `c4a_bundle_off`
  control. Corollary: **every null in the 2026-09-19 regularization sweep was measured on the old,
  partly-dead feature set**, so those knobs are re-openable in principle.
- **Config layout: one template for every model** (2026-09-20).
  `config/template.yml` declares every model's keys - the 271 shared ones (paths, `spec_params`,
  `mol_params`, trainer, loss, logging, eval) plus all 258 model-owned ones - so every run
  resolves the same 529-key schema and shared code can read any key without first checking
  `model_type`. `model_type` selects code, not config: from the custom config, else an explicit
  `model_type` argument (a checkpoint's), else `frag_gnn`; `VALID_MODEL_TYPES` in `runner.py`
  only guards against a typo in the value. The file is the shared core followed by one
  clearly-marked block per model, with model keys flat and prefixed; `tests/test_config_templates.py`
  enforces that a prefixed key sits only in its own block and that the six unprefixed exceptions
  (`magma_params`, `magma_dp`, `ann_params`, `ann_fp`, `spectrum_encoder_params`,
  `retrieval_eval_size`) are declared by name.

  This **replaces the per-model split** attempted earlier the same day, which gave each
  `model_type` its own self-contained `config/<model>_template.yml`. The split's only benefits
  were trimmer hparams and foreign-key warnings; its cost was a new failure class - a key moved
  into one model's template while its reader stayed in shared code, breaking every other model at
  runtime rather than at validation time. That fired for `num_log_*` and the `mass_shatter_*`
  sampler keys, plus a cluster-staging variant (validation read sibling templates the job never
  staged) and a latent one where 8 of 10 templates declared `model_type: frag_gnn`, so a
  non-FraGNNet checkpoint merged with a silent custom config became `frag_gnn`. It also cost
  5,133 lines of YAML against 922, with 271 keys duplicated tenfold behind a drift guard that
  covered 28% of them. One schema removes the class: nothing can be present for one model and
  absent for another. The caller-supplied `model_type` now outranks the template default, which
  fixes the checkpoint case; a regression test pins it.

  Kept from the split, because neither depends on the file layout: **recursive validation** of
  custom configs (a typo inside `spec_params` used to be accepted silently) and the `model_type`
  known-set check. Dropped: the inert-versus-unknown distinction and the sibling-template glob -
  with one schema, a key the template does not declare is simply a typo, and a cluster job stages
  one file whose validation cannot differ from the submitting machine's.

  Historical configs were not rewritten: `config/` is tracked experiment provenance. Validation
  surfaces the same 18 configs that set keys no template has ever declared - 14 were already
  rejected before any of this work, 4 are nested typos that only recursive validation catches
  (`config/clip/nist20_spec_mol_{clip,siglip,soft_clip}_pretraining.yml` set undeclared
  `spectrum_encoder_params.*` sub-keys, `config/debug/debug_tiny_new_frag_mol_feats.yml` sets
  `frag_params.pyg_pe_embed_k`). None were edited. Nesting the metadata-conditioning keys into a
  `conditioning_params` section was tried and **reverted** - it invalidates every trained
  checkpoint for no behavioural gain - and the same argument rules out nesting model keys under
  per-model mappings. Five keys no code reads were deleted from the template and from the 203
  configs that set them: `wandb_root_dir`, `nb_formula_normalized_entropy_weight`,
  `group_sampler_type` (a leftover from the rename to `simple_group_sampler_type`, which is what
  `runner.py` validates), `debug_val_loop_trigger` and `debug_bypass_train_val`. The image-logging
  keys `num_log_train_images`/`num_log_val_images` were deleted with their code: both were 0 in
  every template, no config set them, and `SpectrumPL._log_images` was their only reader.
  Directories renamed on request: `config/models` -> `config/models_template` and
  `config/masshatter` -> `config/massshatter`. Validation: full suite green; `debug_tiny_merged`
  and `debug_tiny_mass_shatter` train end-to-end. Still open: which conditioning mechanism
  survives is the [conditioning refactor](plans/conditioning_refactor_plan.md)'s question, not
  this layout's.
- [Model improvement roadmap](plans/model_improvement_plan.md) remains the canonical source
  for research priorities and acceptance criteria. Its banner is dated 2026-07-24; reconcile
  its proposed next action with current code and artifacts before resuming it.
- [Documentation index](README.md) lists live, proposed, landed, and closed work, including
  MASShatter plans. Those are recorded statuses, not verification of current execution.
- Read linked [findings](findings/) before reviving a rejected or closed experiment.

No training processes, scheduler jobs, checkpoints, or external run statuses were audited during
workflow adoption. No new research priority or scientific conclusion is inferred here.

## Next task

The [FastGNN proposal option](plans/masshatter/masshatter_fast_gnn_proposal.md) is implemented as an
explicit alternative to cut-mask proposals, preserving the default and checkpoint shapes.
Full suite: 3,451 passed, 96 skipped, 2 xfailed; three separate GPU checks and a real-data
forward/backward smoke passed. The linked plan supplies the config and score semantics.
The connected-fragment option now aligns search and scoring with boundary-cut/internal-intact
probabilities from the updated prototype; see the linked plan for changed checkpoint semantics.
Three merged MA/MI configs (128/256/512 fragments and matching beams) and a generation-only
Vulcan launcher with three-hour jobs are prepared in the linked plan.
The [connected-fragment audit](plans/masshatter/masshatter_fast_gnn_proposal.md#connected-fragment-audit-2026-09-17)
found two selection issues now fixed in the working tree: stable beam ties remove padding
dependence, and complements are scored before output-budget pruning. Permanent regressions
and model/backend checks: 220 passed, 11 skipped in the sandbox; the six proposal CUDA checks
subsequently passed outside it. The [real-graph speed audit](findings/masshatter/connected_fragment_beam_speed_2026_09_17.md)
measured candidate time 238.7 → 732.1 ms and forward/backward time 744.4 → 1,252.3 ms at batch
128 / beam 256 on RTX 2070 SUPER. Complement connectivity dominates; peak whole-step allocation
stays near 2.83 GB. Next: reduce unnecessary connectivity checks without changing selection,
then measure L40S throughput and coverage. No production code changed during the speed audit.
All four connected-fragment arms now use `mass_shatter_fast_gnn_max_boundary_cuts: 5`: at 3 the
boundary limit, not the fragment target, was the binding candidate budget, so the 256 and 512
arms filled only 63% and 40% of their budgets and ran at roughly half the `cut_masks` candidate
count. Template and checkpoint-migration defaults stay at 3.
On real NIST20 batches the beam is **not** a throughput improvement: 1,024 ms median against
70 ms for `cut_masks` with `candidate_backend: cuda` and 157 ms for its CPU backend, at batch
128 and comparable fragment counts. Batch cost is set by the batch maximum heavy-atom count,
whose median is 94 at batch 128, and 77% of the time is the per-growth-round complement
reachability loop introduced by the audit fix. See the
[real-precursor measurement](plans/masshatter/masshatter_fast_gnn_proposal.md#real-precursor-throughput-measurement-2026-09-17)
for the profile and the proposed restructuring. Next: screen this arm on accuracy only, or
restructure complement connectivity before making any speed claim; no training run was launched.

MASShatter training speed (2026-09-17): the
[s3gy1ro4 audit](findings/masshatter/masshatter_training_speed_2026_09_17.md) identifies monitoring-only
channel attribution and base CE as removable overhead. A matched fast config and reusable
benchmark are prepared; batch-128 local GPU forward/backward throughput improved by 1.40×.
Scientific losses and checkpoint shapes are unchanged; the two diagnostic families are disabled
in validation too. Loss/gradient equivalence and checkpoint compatibility passed. The small
attention-layout experiment was removed after failing to demonstrate useful whole-batch gains.
A batch-256/accumulation-1 variant is also prepared, preserving effective batch 256. It fits the
local sampled GPU workload (7.10 GB) but shows no throughput gain over batch 128; see the same
finding for matched measurements. Next: compare 128×2 and 256×1 on L40S and verify resolved resume
flags; no remote run or cluster job was changed. The requested 512×1 config is also prepared
with LR unchanged; its effective batch doubles to 512 and its GPU validation remains pending.

MASShatter proposal loss (2026-09-18): the per-spectrum proposal coverage loop is now a
batched whole-batch match, measured 118.6 ms to 2.4 ms per call locally at production shapes
with the previous loop kept as the test reference; see the same
[training-speed finding](findings/masshatter/masshatter_training_speed_2026_09_17.md). No end-to-end
training-step measurement was run.

The [candidate-backend plan](plans/masshatter/masshatter_candidate_backend_plan.md) is active. CPU numeric
construction now releases the GIL and shares a persistent thread pool; local full-training
throughput improved about 9.3% with exact candidate/gradient checks. The score-free selector
specialization was removed after failing its throughput gate. Membership compaction and streaming
are deferred by the current projections. The same [speed finding](findings/masshatter/masshatter_training_speed_2026_09_17.md#cpu-implementation-and-full-training-measurements-2026-09-17)
records results and limits. Batch256 fits locally; batch512 runs out of memory on the 8 GB GPU.
[Three-hour L40S benchmark commands](guides/benchmark_masshatter_cpu.md) are prepared; no jobs were
submitted. CPU remains the default. The exact optional CUDA backend now passes local candidate,
model-gradient and checkpoint checks, with 19–23% higher full-training throughput than
improved CPU in the final 128/256 matrix. Explicit CUDA configs are available for all three batch sizes; see the same guide
for supported modes and compiler requirements. Final CPU full suite: 3,422 passed, 93 skipped,
2 xfailed; separate CUDA suite: 118 passed. Batch512 hits the local 8 GB GPU limit with either
backend. Next: confirm throughput, memory and native compilation on L40S before production promotion.

Merged MASShatter arms for the candidate backend (2026-09-17): six configs under
`config/massshatter/ma_mi_merged/` give merged arms at batch 128x2 / 256x1 / 512x1 on both the CPU
and CUDA candidate backends. They are derived from the unmerged fast arms but are **not** a
controlled A/B on merging: by user decision `spec_params.ace` and `mass_shatter_include_cmf` are
both off, because no merged MASShatter config has ever had a live CMF channel together with dual
CE (`context_ablation`/`smarts` set `ace: True` but sit on the `[M+H]+`/FT-only `nist20_inchikey`
split, so their Na/Cl adducts are absent from the data; the one merged config with live CMF is
NCE-only). Merging is therefore confounded with CE conditioning and the CMF channel; isolating
merging needs unmerged siblings with the same two flags off, which do not exist yet. Epochs are
378/630 rather than 60/100 because merging by acquisition `group_id` collapses train 287,721 ->
45,639 rows and val 96,153 -> 15,257 (6.30x), so the row budget matches (28.75M against 28.77M).
Merging does not strip adducts: 5,993 `[M+Na]+` and 748 `[M+Cl]-` of 45,639 merged train rows, no
group mixing adducts, instruments or frag modes. A merged row conditions on the mean-pooled
acquisition CE ladder (median 4 readings, max 86), not a single energy. Validation: all six
resolve through `load_config` with exactly the intended delta; the merged dataset constructs on
real NIST20 data; `tolerance_consensus` merging of whole multi-spectrum groups produces consensus
peaks; the CUDA config builds a model resolving `candidate_backend` to `cuda` (8,427,349 trainable
parameters against 8,434,087 with ACE/CMF on); 186 config/merge/fast-GNN tests passed. Merged and
unmerged validation cosine score different targets and are not comparable. Not measured: any
training step, batch-512 memory, epoch wall time, and the 6.3x larger fixed per-epoch overhead of
a 630-epoch run. See the [merged arms guide](guides/benchmark_masshatter_merged_control.md).
One-hour L40S jobs are the right size for the backend comparison but not for training: merged
rows fall 6.30x but peaks per row rise 31.9 -> 51.9, so total peak work falls only 3.83x, putting
a merged epoch at roughly 5.3-6.5 min at batch 128 against the measured unmerged 33.52 min
(estimate from row/peak counts, not a measured merged epoch). One hour yields about 8-10 epochs,
so 6 one-hour jobs cover the CPU-vs-CUDA throughput/memory comparison, while 630 epochs would
need 63-79 jobs per arm chained through `--dependency=afterany` with `j=` resume. The guide
carries the arithmetic. No job was generated or submitted.

Live fast-run follow-up (2026-09-17): the existing L40S runs `dny62x0i`, `h0ibnndb`, and
`du6cr9dx` still report the earlier separate candidate timing fields and no backend setting.
Four CPU selection threads are enabled, but the new parallel topology/CUDA implementation is
not reflected in those runs. Monitoring-only fast128 reduces epoch time from 44.5 to 33.5 minutes;
256/512 take 32.1/31.8 minutes, so larger batches alone leave the CPU bottleneck. The old CPU
batch512 run does fit L40S (about 14.76 GB median sampled allocation); new-backend L40S acceptance
remains open. See the [live follow-up](findings/masshatter/masshatter_training_speed_2026_09_17.md#live-fast-run-follow-up-earlier-candidate-code-still-running-2026-09-17).
Next: launch a fresh job from transferred updated source; running job-local installs will not
pick up these working-tree changes. No remote jobs were modified by this read-only audit.

Unmerged generalization audit (2026-09-16): the
[27-run W&B census](findings/masshatter/masshatter_unmerged_overfitting_2026_09_16.md) supports
widening train/validation loss gaps in mature runs, with modest continued cosine gains.
The current `s3gy1ro4` run was still improving at epoch 8. No train cosine is logged.
This audit corrects the anchor provenance below: `ef7fl5c6` is **merged**, not unmerged.
The claimed architecture-invariant 0.688–0.693 band excludes many of the 27 runs.
Current `GroupSampler` does resample through the epoch hook; MASShatter's inherited
optimizer setup ignores cosine scheduler options. Next: matched evaluation-mode train/val
checkpoint metrics, then separate fixed-objective LR and regularization controls.
Documentation and a sampler probe only; no training or model/config changes.

Unmerged MASShatter CMF anchor prepared (2026-09-16): the
[improvement plan](plans/masshatter/masshatter_improvement_plan.md#corrected-unmerged-cmf-anchor-prepared-2026-09-16)
records a new config,
`config/massshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_unique256_enum4096_gine_gsa384_bondcentric_a2_b3_transformer_full_context_dual_ce_proposal_selection_only_inst_tol.yml`,
and a generation-only launcher
`scripts/manual_slurm/vulcan/submit_masshatter_unmerged_cmf.sh`. It is the lead unmerged arm
(`ef7fl5c6`, 0.7515) plus dual CE, `inst_types` reduced to the FT/QTOF actually present in the
split, channel diagnostics, and `ace_max: 300` (the template's 100 clamps 3.49% of this split's
spectra into one saturated embedding row once ACE is on); a resolved-config comparison shows
`spec_params` and `ace_max` as the only differing keys. Two conditioning differences against the
FraGNNet MA/MI reference are left in place as unmeasured treatments rather than defects:
`mass_shatter_context_in_gnn` and `mass_shatter_smarts_templates`, both `False`. The prepared
merged `context_ablation` pair is the place to settle the first. A two-arm proposal-warmup
pilot (`_warmup_e0_{2,4}.yml` beside the anchor, launcher
`scripts/manual_slurm/vulcan/submit_masshatter_unmerged_cmf_warmup.sh`) supplies the bounded
matched pilot that step 3 of the improvement plan asks for; each arm differs from the anchor by
`mass_shatter_proposal_only_until_epoch` alone, and the anchor is the no-warmup control. A real
four-epoch smoke of `e0_2` crossed the boundary with validation cosine flat through the frozen
epochs and rising once the intensity head unfroze. This is the first config to train the 2026-09-14 CMF alignment, because only its
`[M+Na]+`/`[M+Cl]-` spectra (38,156 and 7,820 of 480,002) have a non-degenerate CMF ladder. The
twelve-hour allocation and `j=` resume address the wall-time half of the convergence blocker that
holds up [Arm C](plans/masshatter/masshatter_ion_h_channel_selection_plan.md). `filter_subformula_peaks`
stays off so the target spectra do not change alongside the candidate universe. Validation:
config resolved, both arms constructed, generated job script passed `bash -n`, and one full
subsampled epoch ran end-to-end on the local GPU (exit 0, `val_cos_sim_0.01_epoch/mean` present,
`last.ckpt` written). No Vulcan job was generated or submitted from this workstation, which has
no `sbatch`. Next: run the launcher on the Vulcan login node, `sbatch` it, and record the
scheduler and W&B run IDs in the improvement plan.

Subformula filter hooked up to MASShatter (2026-09-16): the merged Cl/Na fix reaches MASShatter
through the shared `BaseDataset` cleanup with no code change, but it was **inert for every
MASShatter config** — all of them use `data/split/nist20_inchikey`, which is 209,698 `[M+H]+`/FT
spectra with no Na, Cl, QTOF, or IT. Added
`config/massshatter/merged/subformula_filter/nist20_ma_mi_merged_independent_jss_warmup_e0_15_dense_subformula.yml`,
identical to its `[M+H]+` sibling except for `split_dp: data/split/nist20_ma_mi_inchikey`,
`prec_types` (four adducts) and `inst_types: [FT, QTOF]`, so the two remain a matched A/B on adduct
scope. MASShatter needs no DAGs, so the split change costs no preprocessing. The
[subformula finding](findings/subformula_filter_ft_qtof.md#clna-budget-fix-measured-on-the-multi-adduct-split-2026-09-16)
records the measured pre-fix/post-fix delta (+90 peaks of 62,702, every adduct nonnegative),
per-adduct retention at instrument tolerance, and the model/filter agreement on CMF validity.
Validation: 344 passed, 2 xfailed across the filter, MASShatter, dataset, and split-filter suites;
config resolved through `load_config`; cleanup run end-to-end on 2,620 real multi-adduct spectra
with zero emptied spectra. No training run was launched and no checkpoint was evaluated.
Next: launch the new config as the matched multi-adduct arm, or first measure oracle coverage on
the Na/Cl population. Note the existing 4-adduct MASShatter configs (`context_ablation`, `smarts`)
declare Na/Cl in `prec_types` but sit on the `[M+H]+`-only split, so their extra adduct slots and
CMF channels have never been exercised.

Inference path consolidation merged from `feiw_v2` (2026-09-16): the
[inference step consolidation plan](plans/inference_step_consolidation_plan.md) is LANDED on that
branch and is now merged here. It collapses the three `SpectrumPL` inference entry points into one,
deletes `resolve_pred_ints` and its `step_untransformed` flag, and removes
`GreedyInferenceBatchSampler` in favour of the lazy `IterableSpecMolFragDataset`. The same merge
brings the subformula-filter fix for Cl and Na adducts, HDF5 output compression, and the
`drop_empty_spectra` prediction-only fix. Training and loss paths keep `pred_logprobs` and were not
touched. Two contract changes from `feiw_v2` win over this branch's prior behaviour: the exporters
now rescale each spectrum's base peak to 1000 on top of the L1-normalized `pred_ints`, and
`run_inference.py` no longer offers `--save_format parquet`. Two MASShatter tests that pinned the
old exporter scale and the parquet writer were updated accordingly; this branch's `requires_dags`
guard and its `pred_intensities` output contract are preserved. Validation: 3,234 passed, 1 skipped,
2 xfailed; Ruff on the resolved files matches the `feiw_v2` baseline. Next: a real-checkpoint
inference smoke check; no GPU run or evaluation was performed for the merge.

Candidate attention topology audit (2026-09-13): the
[audit section](findings/masshatter/masshatter_code_audit.md#candidate-attention-topology-audit-2026-09-13)
records seven findings on the uncommitted `full` candidate-attention arm, all fixed in the
working tree: a hyperparameter fallback default plus a missing `get_pl_hparams` migration,
corrected A/B parameter counts, an unstated capacity confound, a missing unique-fragment guard,
a pre-existing CUDA bfloat16 `index_add` crash, two documented degenerate head inputs, and a
per-layer host sync in `_efa_linear_attn`. Bucketed attention masking was verified exact for both
topologies, and `full` is cheaper only up to about 256 candidates. Validation: full suite plus
274 focused tests, Ruff check and format on changed files, and construction of both A/B arms.
A pre-rebuild full-suite run gave 92 failed, 2,873 passed; every failure was the then-stale
multi-cut Cython extension, all carrying `compute_ccs_multi_cut()` unexpected-keyword errors
for `max_ring_cut_steps` or `ring_open_mask`. Both extensions were rebuilt in place at
2026-09-13 18:12, and `mass_shatter_core` was force-rebuilt at 22:38 to a byte-identical
binary, confirming the 18:12 build already matched its source. All four extensions are now
current and the full suite passes: 2,975 passed, 1 skipped, 2 xfailed.
Next: decide whether to remove the degenerate `component_residual_head` and constant mask-count
feature, which would invalidate existing MASShatter checkpoints and so needs explicit
authorization. No training ran and no run was launched for this arm.

D4 oracle verification (2026-09-13): the
[subformula-filter finding](findings/subformula_filter_ft_qtof.md#saved-subformula-filtered-d3d4-comparison-verified-2026-09-13)
records reaggregation of the saved D3/D4 artifacts. MASShatter's roughly 94% coverage is strong
relative to that unmatched D4 reference; the remaining 6% is not established cut-scoring headroom.
No preprocessing, DAG generation, or model inference reran. Next: compare current MASShatter
support-oracle and predicted cosine on identical targets. Bond-cut-first remains the requested
diagnostic order, with training conditional on useful support or efficiency headroom.

MASShatter H-support ablation (measurement 2026-09-14; code clarification 2026-09-15): the
[ion/H channel plan](plans/masshatter/masshatter_ion_h_channel_selection_plan.md#h-support-ablation-measurement-2026-09-14)
now records the matched 2026-09-08 gated / supervision-only / no-support arms read from W&B. No
trained arm exercises channel selection: retained channels equal valid channels in every arm
(about 8.5 per fragment against roughly 1.02 attributed), so the hard-selection stage has never
been measured. The soft support gate shows no spectral or weighted-precision benefit in either
data variant, and all dense cosine differences fall below one per-epoch standard deviation of a
single seed. The plan now records clone-initialized self-gating as an untested explanation for
the gate null, and clarifies that the six-run scope tests only H support (one valid ion mode,
at most nine CRF H channels). The support BCE does concentrate 95% of per-fragment channel mass into about 11%
fewer channels without moving cosine. Next: disable the gate in the H-support configs, verify the
`required_ion_h_top_k_95`/`_99` metric that reports identical means, and recover one cluster
checkpoint so the Phase 2 selection sweep can run; W&B holds no checkpoint artifacts for these
runs. Documentation only; no training, evaluation, or code change.

Joint selection objective (2026-09-13): the
[ion/H channel plan](plans/masshatter/masshatter_ion_h_channel_selection_plan.md#joint-fragmentchannel-selection-loss-experiment-2026-09-13)
now records a proposed single selection loss that couples fragment proposal and
ion/H support probabilities while retaining the independent spectral intensity loss.
The current JSS + proposal loss + support BCE configuration is the control. Next:
check joint-loss gradients and the all-channels-open failure mode on synthetic cases,
then compare on matched validation support and intensity metrics. Documentation only;
no implementation, config change, or training run.

MASShatter SMARTS review (2026-09-13): the [set/literature review](findings/masshatter/masshatter_smarts_review_2026_09_13.md)
confirms 26 rules, 19 legacy rules without enforced ionization guards, and a pyrimidine
gap in the carbamate family. Positive-ion oxygen-transfer and prazole-sulfone candidates
are supported literature leads, not measured coverage gains. Validation: 75 SMARTS tests
passed on CPU plus registry/matching probes. Next: measure existing rules' incremental
coverage on filtered validation peaks before adding narrowly mapped rules. No model edits
or training ran.

MASShatter proposal review (2026-09-13): the [proposal review](findings/masshatter/masshatter_proposal_review_2026_09_13.md)
prioritizes support diagnostics before encoder expansion and separates established loss/routing
limitations from exploration, diversity, and mixture hypotheses. A fresh 155-run W&B query
and eight full validation histories support selection-only in a close same-seed historical
pair: +1.54 percentage points candidate weighted recall and +0.01261 cosine at epoch 142.
A separate larger-intensity-head pair gains 0.00897 cosine at epoch 98 despite slightly
lower coverage. Recent matched JSS/KL results are close and do not establish a KL advantage.
Next for the intensity stage: preserve selection-only and diagnose fragment-versus-H ranking
on fixed support before another head or loss comparison; see the review for provenance limits.
A live W&B check of three
recent merged U256/E1024 histories confirms only 5.5–7.0% reach U256 at their latest logged
validation epochs. Two existing unmerged pairs match in logged configuration except E and
labels: E4096 adds about 43–46 fragments but only 0.66–0.67 percentage points of candidate
weighted recall at common epochs. A subformula-filter query found 14 runs, all U256/E1024;
late-run coverage is about 94%, with no filtered E4096 counterpart found. No new tests or
experiments ran. Next: inspect remaining
missed intensity and support-oracle headroom before proposing another budget run. The proposal review
is present but untracked, as is its linked loss review; include both in the eventual documentation
change set with these indexes. No files were staged or committed.

EBM modeling proposal (2026-09-13): the user selected both approaches, with
[bond-cut EBM modeling first](plans/masshatter/masshatter_bond_cut_ebm_plan.md), then spectrum reranking.
The [MASShatter program](plans/masshatter/masshatter_energy_based_model_plan.md) records this ordering and
the spectrum setup delta. Next: select a checkpoint and validation subset, measure missed-intensity
and finite-mask-pool headroom, then evaluate pairwise cut energies against search-only, unary,
random, and mixture controls. Preserve probability contracts; the existing exact heap guarantee
does not extend to pairwise energies. No implementation or experiments started for this program.
The spectrum stage retains its separate shared protocol and null-adjusted gate.

Loss setup review revalidated (2026-09-21, `551a64ae`): the
[current assessment](findings/loss_setup_review_2026_09_13.md#current-assessment-tighter-objectives-2026-09-21)
reproduced all four FraGNNet defects on the starting source. Two vectorized-JSD defects are now
fixed; the current focused suites report 393 passed and 6 skipped. CPU probes reverified. A
separately named MASShatter
`conditional_cross_entropy` path is now in the working tree while historical `cross_entropy`
remains unchanged; both explicit CE variants are logged. FraGNNet's vectorized Hungarian JSD now
uses the scalar JS-gain assignment and supports CPU fallback backward; regression probes agree
within 1e-6. MASShatter direct losses and vectorized FraGNNet JSD share an explicit sparse-support
alignment primitive while retaining their different matching policies. No training was run.
Explicit zero-safe `hellinger_distance` is now wired into MASShatter's closest/Hungarian direct-loss
path and FraGNNet's sparse binned path. Focused value and backward tests include exact structural
zeros and reproduce the suppressed-correct-peak probe gradient of `-0.004046`; no Hellinger training
run has been performed. The two auxiliaries are now implemented and default off:
`spectral_leak_loss_weight` penalizes the
normalized intensity outside every true tolerance window independently of Hungarian assignment,
and `spectral_ce_auxiliary_weight` applies historical coverage-weighted sparse Gaussian CE. For
independent MASShatter output, candidate scores are L1-normalized before CE. Both terms are restricted
to Hellinger configurations and logged separately; their coefficient sweeps remain unrun.
Next: correct the formula-validity contract and settle OOS semantics, then measure
supported-peak ranking and per-term gradients before a targeted ranking experiment.
The recorded campaign does not establish a benefit from a generic JSD-to-KL replacement.

Second-pass loss audit (2026-09-21): the working-tree loss changes above were audited as changes
rather than against the older defects; see the
[second-pass section](findings/loss_setup_review_2026_09_13.md#second-pass-audit-of-the-working-tree-2026-09-21).
Four defects in them are fixed: the vectorized Hungarian JSD returned NaN for a spectrum with
observed but no predicted peaks (the emptiness guard is per chunk, and `551a64ae` scored the same
input 0.0, a perfect score for an empty prediction); the infeasible-pair sentinel and the SciPy
backend broke under float16/bfloat16 where a dropped `.float()` had been load-bearing; the leak
term inferred `batch_size` from observed peaks, so a spectrum without observed peaks shortened the
result and broadcast silently onto a per-spectrum loss; and the leak term ran an unbatched
per-spectrum loop with three to four device synchronizations per spectrum. The leak term is now
the chunked padded form (8.9x faster at batch 512 on CPU, values identical) and takes an explicit
`batch_size`. **Compatibility: `jss_hun`, `jss_hun_sqrt`, `jss_hun_np`, and `jss_hun_inst_tol` move
with the gain-matching change** — 40% of 300 random padded spectra change, mean +0.0165, maximum
+0.2653, never worse — so those metrics are not comparable across it. `cos_hun` is unaffected.
Cleared on measurement: the shared alignment primitive is bit-identical to the code it replaced
(`max |new - old| = 0` over 180 spectra, three objectives, both policies), vectorized and scalar
JSD agree to 6.3e-07, sparse Hellinger matches a dense reference to 1.8e-07, and conditional CE
agrees across the sequential and vectorized CE paths. `1 - BC` is the squared Hellinger distance
`H^2`, recorded in the docstrings and config comment. Validation: full suite **4068 passed,
1 skipped, 2 xfailed**; Ruff and `git diff --check` clean; CUDA assignment unavailable, so the
CUDA backend and any training remain unrun.

Both `hellinger_distance` paths were checked for batching and are batched: MASShatter's objective
reduces inside the `loss_batch_size` chunk loop of `sparse_peak_list_distance`, and FraGNNet's
binned `sparse_hellinger_distance` has no Python loop. The gap was coverage: the objective had
been omitted from the four parameterizations in `tests/test_mass_shatter_peak_list_distance.py`,
including the per-spectrum loop-reference harness, so its chunked reduction had never been
compared against a reference. The reference now has an independently written Hellinger branch,
the objective is parameterized everywhere the others are (+57 cases), and chunk invariance,
batch-composition independence, per-spectrum gradient isolation, and a closed-form shared-bin
case were added for both paths. Three mutations confirm the coverage bites.

Original loss setup review (2026-09-13): the [loss review](findings/loss_setup_review_2026_09_13.md)
reproduces new FraGNNet vectorized-JSS correspondence/backward defects, ignored OOS
supervision under JSS, and invalid formula slots stealing support targets. MASShatter
probes identify weak JSS recovery gradients, correspondence/support ambiguity, proposal-loss
coverage limits, and a no-warmup gradient-routing confound. No implementation changes or
training ran; 195 focused tests passed on CPU. Next: fix the verified FraGNNet contracts,
then inspect MASShatter gradients/coverage before an isolated matched JSS/KL comparison.
The prior audit's three FraGNNet loss/explanation fixes and MASShatter precursor-pair
rejection are already present in the reviewed source; their earlier pending status below
is historical. The W&B screen report is missing from this checkout; its historical summary below
is not reverified evidence.

Correctness audit (2026-09-12): the [combined model audit](findings/fragnnet_masshatter_audit_2026_09_12.md)
records reproduced checkpoint-loading, HDF5 cutoff, FraGNNet loss/explanation, and shared
test-metric defects, plus silently ignored MASShatter precursor-pair supervision.
No fixes were applied. The existing local multi-cut extension is
stale; an isolated rebuild from current source yielded 2,876 passed, 21 skipped, and
2 expected failures for the full suite. Next: address checkpoint loading and test-metric
initialization, then the remaining counterexamples. Pre-existing SMARTS warmup edits
were preserved; CUDA and real-data scientific behavior remain unverified.

W&B review (2026-09-11; historical summary, source report unavailable here): the recorded screen
superseded older “no runs launched” statements. `docs/findings/wandb_screen_2026_09_11.md`
is absent from this checkout; recover the report before using these comparisons as evidence.
JSS warmup 0–4 leads the single-seed filtered screen at 0.80099 best validation binned
cosine; CE remains close and finished sooner. Support supervision helps early training;
an additional soft-gate gain is unresolved. GNN context completed, but its matched
heads-only comparison is missing from the inspected runs. SMARTS warmup failed at the
known construction guard. Next: matched-seed confirmation and the missing controls;
recover the missing report for metric definitions, run links, and provenance limitations.

SMARTS merged readiness (2026-09-09): the [readiness audit](findings/masshatter/masshatter_code_audit.md#smarts-merged-run-readiness-2026-09-09)
passed 494 CPU tests (5 skipped) and a synthetic SMARTS + merged NCE + GNN-context
forward/backward probe. The current merged configs are blocked by their proposal-only
warmup, which explicitly rejects SMARTS. The requested no-warmup SMARTS config is now
prepared and linked in the audit; full model construction and intended-only resolved config
differences passed. Next: perform a real-data GPU smoke check. No runs launched.
The subsequently requested [SMARTS warmup config](../config/massshatter/merged/smarts/nist20_merged_full_metadata_context_gnn_heads_smarts_warmup_e0_15.yml)
is also prepared, with proposal-only epochs 0–15. The SMARTS/proposal-only-warmup
construction guard is now removed and validated (2026-09-11): see the
[guard-removal finding](findings/masshatter/masshatter_code_audit.md#smarts-proposal-only-warmup-guard-removed-2026-09-11)
for the new PL-level gradient-isolation regression tests. Construction now succeeds.
Next: a real-data GPU smoke check before launching this arm.

Context-conditioning screen (2026-09-08): the user selected controlled experiments before a new
layer. The [conditioning plan](plans/conditioning_refactor_plan.md#current-controlled-screen-2026-09-08)
now selects the full-metadata, dual-CE MASShatter mask1024 config as the conditioning baseline:
the user reports that NCE + ACE + adduct + instrument + mode already helps in existing runs.
The prepared NCE-only variant is an optional diagnostic outside the next run queue. Two new
FraGNNet configs complete the existing full-metadata routing pair with output scalers disabled.
Follow-up: the user requested context at the merged MASShatter GNN input. The optional
`mass_shatter_context_in_gnn` path is now implemented, default off with legacy-checkpoint
migration. A matched full-metadata merged heads-only versus GNN+heads pair is prepared;
existing experiment configs are unchanged. Validation: 182 passed, 5 skipped, both merged
configs constructed, intended-only resolved diffs and Ruff passed. No runs launched.
The [pre-run design audit](findings/masshatter/masshatter_context_design_audit.md) now confirms homogeneous
groups, but all selected rows are `[M+H]+`/FT/HCD. It found ACE saturation at 100 eV in 21.42% of
selected training groups and row-order-dependent ACE selection for same-NCE replicates. Actual
GINE-EFA forward/backward checks at epochs 15/16 were finite; no quality evaluation ran.
CE fixes are now implemented: the new pair uses ACE up to 300 eV and order-invariant weighted
CE aggregation, with both models consuming the weights and legacy behavior preserved. The
[audit follow-up](findings/masshatter/masshatter_context_design_audit.md) records regression and source-data
checks. GNN initialization remains unchanged. Next: a real-data GPU smoke check before training.

FraGNNet formula-support crash follow-up (2026-09-07): W&B runs
[kvrlaxof](https://wandb.ai/frag-gnn/frag-gnn/runs/kvrlaxof) and
[71qtthvw](https://wandb.ai/frag-gnn/frag-gnn/runs/71qtthvw) both failed with infinite
loss on batch 0 before backward. Local reproduction shows that positive support targets for
unreferenced formula slots can overflow the binary loss: their logits retain the float32
minimum sentinel. The working-tree fix in `fragnnet_model.py` derives support validity from
valid joint-state contributors and zeros invalid support logits. Tensor shapes, parameters,
configs, and spectral intensity computation are unchanged. Regression coverage verifies both
attribution methods, finite gradients, max pooling, and empty support. Validation: 40 focused
tests passed on CPU; Ruff and whitespace checks passed. No full training or validation-set
evaluation ran. Next check: run a short cluster training smoke test with the patched package,
checking individual loss terms if failure persists. The exact failing remote batch was not
replayed, so this is a reproduced defect and candidate crash fix, not a confirmed remote cure.

For the next requested implementation or experiment, verify the relevant plan against code and
available artifacts. Link that task here when it spans sessions, and record its next check and
blockers in the plan. Do not duplicate the roadmap or copy its full backlog here.

## Local context caveat

The ignored `.claude/` entry point and six rule files now link to the shared guidance.
These local pointer changes are not included in Git diffs. It still contains a legacy architecture inventory;
its state directory had no files during this audit. Shared records above are the portable
entry points. Local permissions, commands, and skills were not migrated or audited for runtime
behavior. Their presence is not evidence that another checkout has them.

## Cluster submission knowledge

The [Fir/Vulcan submission guide](guides/submit_cluster_runs.md) now provides the canonical
procedure linked from INSTALL.md. Commands and profiles were checked against repository source;
no cluster connection, submission, or training validation was performed.

Connected-fragment beam review (2026-09-17): the
[reuse and equivalence finding](findings/masshatter/connected_fragment_beam_reuse_and_equivalence_2026_09_17.md)
records that the beam objective is `log P(mask = ∂f)` minus the complement's intact term, so it
matches `cut_masks` up to replacing the charge head with a fixed prior, ring-reconnection terms
that vanish for induced-acyclic fragments, and exact-versus-approximate search. Nine of about
thirteen search mechanisms already have MASShatter counterparts; the dominant measured cost,
dense-reachability complement connectivity, is the one with an existing union-find replacement
on CPU and CUDA, which would also make the half-depth-plus-mirror growth free rather than paid
for. The finding also flags likely per-iteration host synchronization in the reachability loop.
The matching `cut_masks` inefficiency is mask-space budget: the unmerged arm enumerates 4096
masks for a 256 unique-fragment target, and that redundancy is already logged as
`unique_components_per_mask` and `duplicate_component_pathway_fraction`, so the efficiency
comparison can be answered from existing run artifacts without a new benchmark.
Source and algebra review only: no measurement, training, test run or code change. Next: the two
recorded gap checks (ring-bond probability histogram and implicit-prior versus charge-head split)
before any further beam optimization or scientific comparison.

Beam direction set (2026-09-17): keep the connected-fragment beam as a proposal-only
enumerator and restore a learned charge component, reconciling the prototype's two scores
toward its search form (the isolating mask's log probability) rather than its output form.
This removes the size/topology prior, keeps the chemically correct ring penalty through
boundary-cut count, makes complements score-symmetric again and so removes most of the
447 ms complement cost. The one structural item is emitting each mirror pair as two
components of one mask so the existing charge split and unique-component marginalization
apply unchanged. Recorded as PROPOSED in
[the FastGNN plan](plans/masshatter/masshatter_fast_gnn_proposal.md#proposed-proposal-only-beam-with-learned-charge-2026-09-17);
not implemented. The gate has now been run: at matched fragment target on identical bond logits
the beam retains the smaller side of a cut about 5:1 (fraction above half 0.167 versus 0.590),
proposes about four times fewer large fragments (0.079 versus 0.319 at or above 0.75 of the
molecule) and 1.5 times more acyclic fragments. Raising `max_boundary_cuts` from 5 to 20 leaves
the distribution unchanged and raising the fragment target makes the bias worse, so neither the
cut cap nor the budget explains it. Bond logits were synthetic; the checkpoint-logit
confirmation was not obtained. Measurements and limits are in the same finding. A budget-exhaustion run also separated ranking from coverage: fragments at or above 0.75 of the
molecule saturate near a fixed count as the budget grows 9.4x, because anything above half the
molecule arrives only as a connected mirror, so a connected fragment with a disconnected
complement is unreachable. That is a coverage hole, not a ranking effect, and the same scoring
change plus union-find closes it. The coverage-per-budget gate then closed the question against the beam. Scoring every
reference fragment by the objective the redesign would adopt, the beam's captured probability
mass saturates near 0.70 across an eightfold budget increase while `cut_masks` reaches 0.983,
and the beam costs roughly twenty times more per fragment on CPU. Decomposition by fragment size
locates the ceiling: the beam is complete below half the molecule (1.000 and 0.999 by count) and
reaches only 0.109 of fragments above 0.75 of the molecule, the band holding 47.2% of all mass.
Large fragments from a single bond cut are reachable because their complement is connected;
multi-cut large fragments generally leave a disconnected remainder and are missing by
construction. This is architectural, not a scoring artifact, so the proposal-only redesign would
not lift it. Repeating the coverage scan with **trained** bond logits from checkpoint `zlv0bli8` (epoch 41)
reproduces every direction and revises the magnitude: captured mass is flat at 0.8695 across a
fourfold budget increase against 1.0000 for `cut_masks`, so the cost of the limit is about 13% of
probability mass rather than 30%. A confident trained model concentrates mass on single-cut
masks, whose large fragment has a connected complement and is reachable; the multi-cut large
fragments remain missing. The pre-fix and post-fix beams emit identical fragment sets in 18 of 24
configurations, differing by one or two fragments in 4,001 elsewhere, so the 447 ms complement
pass buys determinism rather than materially different candidates. All measurements are
reproducible through
[scripts/debug/analyze_proposal_fragment_coverage.py](../scripts/debug/analyze_proposal_fragment_coverage.py).
Next: a decision on whether to stop the connected-fragment line; the four prepared arms should
not be trained as configured in any case.
