# MASShatter fragment-marginal selection

> **Status:** CLOSED NULL · **Updated:** 2026-09-24
> Implemented behind `mass_shatter_fragment_selection: marginal`; default behaviour unchanged.
> All five arms ran. Marginal ranking fails both acceptance criteria. Mask-order U128 matches
> the control at 1.65x the speed, and the August U128 penalty does not reproduce. See the
> [tail read](#tail-read-2026-09-24-marginal-selection-fails-and-u128-is-a-free-speed-up).

## Objective and acceptance criteria

Make a small intensity-head budget (K = 128 fragments) keep the coverage of a larger one by
choosing fragments **by their own probability** instead of by the order the mask enumeration
first produces them, and let the proposal loss see every fragment in the enumerated pool
rather than only the kept prefix.

Accept the mechanism when, on the merged `c4a_base` lineage at 128 epochs:

1. `f2` (marginal K128 from a U256 pool) reaches validation cosine within the
   [128-epoch screen gate](../../findings/masshatter/masshatter_architecture_arms_epoch128_2026_09_22.md)
   (±0.005 of the 0.7270 control tail mean) at lower hours per epoch than the control, and
2. `f2` beats `f1` (mask-order U128) on both `val_candidate_wrecall` and cosine.

`f2 ≈ f1` means marginal ranking is no better than mask order at this budget: close the line
as `CLOSED NULL` and keep mask order. `f2` below the control but above `f1` is a partial
repair; read `val_pool_candidate_wrecall` to see whether the pool or the ranking limits it.

## Established facts

### A U128 cap loses coverage, and all of it from multi-cut fragments

Merged NIST20 budget sweep of 2026-08-20 (`masshatter_merged_u{128,256,512}_budget_*`,
E2048, three `base_ce_weight` values per arm as paired replicates, one seed), read at epoch
117 from W&B:

| | U128 | U256 | U512 |
| --- | ---: | ---: | ---: |
| Runs | `swg3j4ep` `jz6ctocr` `7wc3xm3u` | `4roljw8c` `xu1dcl0h` `m2owk5za` | `wqzl4esk` `txhqoe4s` `94zple62` |
| Ranked masks processed | 511 | 1,553 | 1,925 |
| Unique fragments | 125 | 224 | 263 |
| Candidate weighted recall | 0.868 | 0.909 | 0.917 |
| Standalone coverage, precursor / single-cut masks | 0.122 / 0.286 | 0.122 / 0.286 | 0.122 / 0.286 |
| Standalone coverage, 3-cut masks | 0.298 | 0.372 | 0.382 |
| Standalone coverage, 4+-cut masks | 0.250 | 0.392 | 0.415 |
| Duplicate component pathways | 0.908 | 0.956 | 0.966 |
| Best cosine by epoch 117 | 0.7208 | 0.7299 | 0.7334 |
| Epochs inside the 3 h wall / best cosine by then | 175 / 0.7245 | 137 / 0.7308 | 118 / 0.7334 |

U128 − U256 is −0.009, −0.011 and −0.007 in the three paired arms. The precursor and single
cuts are forced ahead of the ranked masks (`_merge_forced_masks` in
[mass_shatter_candidates.py](../../../src/fragnnet/frag/mass_shatter_candidates.py)), so they are
identical across budgets; everything U128 loses comes from fragments needing three or more
cuts. Caveats: August code, the merged target from before the 2026-09-03 `tolerance_consensus`
change, one seed.

### Why mask order is the wrong ranking

Masks are listed in exact descending joint probability (the heap of
[masshatter_plan.md §2.1](masshatter_plan.md#21-exact-top-k-mask-enumeration-under-the-independent-model)),
and the unique-fragment stop keeps the first U atom sets to appear. A fragment F with boundary
bonds ∂F and interior bonds int F first appears at the best complete mask containing it:

```text
score_first_appearance(F) = P_release(F) · Π_{b ∉ ∂F ∪ int F} max(p_b, 1 − p_b)
P_release(F)              = Π_{b ∈ ∂F} p_b · Π_{b ∈ int F} (1 − p_b)
```

The second factor is the probability of the single most likely setting of every bond outside
F. It depends on how decided the rest of the molecule is, not on F, and it halves the score for
every coin-flip bond elsewhere. Summing over the rest instead of maximising gives
`p + (1 − p) = 1` per bond, which is why ranking by `P_release` removes it. Consecutive masks near
the mode differ in bonds far from any new fragment, which is where the 91–97% duplicate pathways
come from.

### Why the proposal cannot learn its way out

`_proposal_coverage_loss` in [mass_shatter_pl.py](../../../src/fragnnet/pl_model/mass_shatter_pl.py)
scores each observed peak by the probability of the retained fragments that explain it and gives
an uncovered peak exactly zero loss (`th.where(covered, …, zeros)`); mask selection uses detached
scores. A fragment past the cutoff therefore never receives gradient. ICEBERG and GLACIER cap a
per-fragment ranking supervised on labelled fragments instead
([GLACIER note](../../notes/glacier_object_detection.md)).

The current fragment probability is itself a partial sum: `fragment_log_probabilities` in
[mass_shatter_model.py](../../../src/fragnnet/model/mass_shatter_model.py) is a log-sum over only the
pathways enumerated before the stop. A fixed-checkpoint cutoff on it cost 0.0075 cosine while
retaining 99.58% of proposal mass
([channel selection findings](../../findings/masshatter/masshatter_channel_selection_findings.md)), so it is not
a usable ranking as it stands.

## Design

### Release score

For every unique fragment of the enumerated pool, computed from the live bond logits:

```text
log P_release(F) = Σ_{b ∈ ∂F} log σ(l_b) + Σ_{b ∈ int F} log σ(−l_b)
```

Interior bonds are counted without a per-fragment bond scan: summing the intact log-probability
of every bond incident to F's atoms counts each interior bond twice and each boundary bond once,
so `Σ_{int F} = (Σ_{a ∈ F} S_a − Σ_{b ∈ ∂F} log σ(−l_b)) / 2` with
`S_a = Σ_{b ∋ a} log σ(−l_b)`.

`P_release` is the exact probability, under the independent-bond model, that F's boundary is cut
**and its interior intact**. For an acyclic fragment that is the full atom-set marginal. For a
fragment containing a ring it is a lower bound, because one ring bond inside F can break without
changing the atom set; the model is trained against this same event, so the ranking and the loss
agree. Charge is not in the score: MASShatter normalises charge retention per mask among sibling
components, which has no closed form, and mask-order selection ignores charge as well.

### Selection

1. The candidate builder is unchanged on both backends: it still enumerates up to
   `mass_shatter_ranked_mask_budget` masks and stops at `mass_shatter_unique_fragment_target`
   unique atom sets, which now defines the **pool**.
2. Pathway marginalization runs over the whole pool as before.
3. The precursor and, with `mass_shatter_include_single_bond_cuts`, every single-cut fragment are
   forced (boundary count ≤ 1), matching mask order, which places those masks first.
4. The remaining `K − forced` slots go to the highest `P_release` fragments that have at least
   one chemically valid channel. Only those K reach the intensity head.

### Proposal loss over the pool

`proposal_candidate_*` outputs carry the whole pool, with `P_release` normalised over the pool
per molecule, so every peak that any pool fragment can explain contributes gradient. The loss
function and its many-to-one log-sum aggregation are unchanged; no per-fragment labels are
introduced.

### Unchanged and out of scope

- The final distribution, intensity head, channel selection and every non-proposal loss.
- `mass_shatter_fragment_selection: mask_order` (the default) runs the same computation as before.
- SMARTS templates: marginal mode rejects them. If they return, force their products outside K.
  Only two runs ever enabled them and the CUDA backend refuses them.
- Correlated cuts (pairwise energies, [bond-cut EBM plan](masshatter_bond_cut_ebm_plan.md)); this
  is its per-fragment efficiency arm.

## Configuration

| Key | Default | Meaning |
| --- | --- | --- |
| `mass_shatter_fragment_selection` | `mask_order` | `marginal` enables this plan. |
| `mass_shatter_selected_fragment_target` | `null` | K in marginal mode; must be ≤ the unique-fragment target, and null otherwise. |

Marginal mode requires unique-fragment mode, `mass_shatter_proposal_selection_only`, one Bernoulli
mixture, `cut_masks`, SMARTS off, a zero fragment-probability floor and
`mass_shatter_share_proposal_across_ev: false`; each violation raises at construction. Checkpoints
that predate the keys load as `mask_order`.

`val_pool_candidate_wrecall` (validation, with candidate coverage metrics on) reports the
weighted recall of the whole pool, the ceiling for any selection from it.

## Experiment

Configs in `config/massshatter/ma_mi_merged/fragment_selection_2026_09_23/`, generated (not
submitted) by `scripts/manual_slurm/fir/submit_masshatter_fragment_selection.sh`. Each is a diff of
`nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_base_s1.yml` at the same seed and 128-epoch box;
the existing `c4a_base_s1`/`s2` runs are the control.

| Arm | Keys vs control | Question |
| --- | --- | --- |
| `f1_mask_order_u128` | U128 | What naive U128 costs on current code. |
| `f2_marginal_k128_pool256` | marginal, K128 | Does marginal ranking recover U256 at K128? |
| `f3_marginal_k128_pool512` | marginal, K128, U512 | Does a larger pool help at the same K? |
| `f4_mask_order_u64` | U64, control monitoring | Budget ladder, added after the early read: is U64 still free, and faster? |
| `f5_mask_order_u128_ctlmon` | U128, control monitoring | `f1` without the coverage metrics: the clean U128 wall-clock. |

`f1`–`f3` run the coverage and cut-bucket metrics; `f4`/`f5` keep the control's monitoring
so their hours per epoch compare directly with `c4a_base`. Generate the ladder with
`submit_masshatter_fragment_selection.sh ladder`.

Read `val_cos_sim_0.01_epoch/mean` as the 118–127 tail mean against the control, as in the
screen read-out, together with `val_candidate_wrecall`, `val_pool_candidate_wrecall` and hours
per epoch.

### Runs and early read (2026-09-23, PROVISIONAL)

Running on Fir, started 18:51–18:55 UTC: `f1` [`uorwhapr`](https://wandb.ai/frag-gnn/frag-gnn/runs/uorwhapr),
`f2` [`u7youmib`](https://wandb.ai/frag-gnn/frag-gnn/runs/u7youmib), `f3`
[`9nxd8y8y`](https://wandb.ai/frag-gnn/frag-gnn/runs/9nxd8y8y). Control: `c4a_base_s1`
`5qws8g9x`, `s2` `6plumx6r`. Read at the latest common validation epoch (27); single-epoch
floor 0.0025, so none of the cosine differences is resolved yet.

| | f1 mask U128 | f2 marginal K128/U256 | f3 marginal K128/U512 | control s1 / s2 |
| --- | ---: | ---: | ---: | ---: |
| Cosine, mean of epochs 23–27 | 0.7103 | 0.7094 | 0.7067 | 0.7093 / 0.7096 |
| Fragments pool → kept | 126 → 126 | 243 → 126 | 406 → 126 | 242 → 242 |
| `val_candidate_wrecall` (kept) | 0.8797 | 0.8796 | 0.8788 | not logged |
| `val_pool_candidate_wrecall` | — | 0.9000 | 0.9106 | — |
| Kept coverage, 3-cut / 4+-cut buckets | 0.202 / 0.117 | 0.207 / 0.083 | 0.209 / 0.078 | not logged |
| Seconds per epoch (train + validation) | 55.1 | 58.2 | 66.4 | 50.1 / 49.2 |

What it says so far:

- **Marginal ranking recovers none of the pool's extra coverage at K128.** The kept set covers
  the same 0.880 under mask order and under release-score ranking, while the U256 and U512 pools
  it ranks from cover 0.900 and 0.911. Release ranking keeps fewer 4+-cut fragments than mask
  order.
- **The U128 penalty of the August sweep is not visible on current code.** August had U128
  0.0095 behind U256 by epoch 27 (0.6835 vs 0.6930, held to epoch 116); here `f1` sits on the
  control. Its coverage deficit is also smaller: about 0.020 against the U256 pool, not 0.041.
- **Speed is confounded.** Only the arms run the coverage and cut-bucket validation metrics, and
  all three are slower per epoch than the control, including `f1` with half the fragments.
  Among the arms, marginal selection costs +6% (`f2`) and +20% (`f3`) over `f1`.

Re-read at the 118–127 tail before drawing a conclusion; the pool-wide proposal loss may still
reshape the ranking.

### Tail read (2026-09-24): marginal selection fails, and U128 is a free speed-up

All five arms finished 128 epochs at seed 397623 on the `fir:h100-3g.40gb` profile the
generator uses for every arm. Each resolves against `c4a_base_s1` to its selection keys, the
`f1`–`f3` diagnostic keys, and inert defaults. Tail means over epochs 118–127 against the 0.7270
control (gate ±0.005). `val_wrecall` is the prediction-side weighted recall; `f4`, `f5` and the
controls do not log `val_candidate_wrecall`.

| | `f1` mask U128 | `f5` mask U128, control monitoring | `f2` marginal K128/U256 | `f3` marginal K128/U512 | `f4` mask U64 | control s1 / s2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| run | `uorwhapr` | `hrty7mdt` | `u7youmib` | `9nxd8y8y` | `3znrt3fk` | `5qws8g9x` / `6plumx6r` |
| cosine | 0.7292 | 0.7277 | 0.7270 | 0.7254 | **0.7180** | 0.7266 / 0.7273 |
| vs control | +0.0022 | +0.0007 | 0.0000 | −0.0016 | **−0.0090** | — |
| fragments kept | 125.8 | 125.8 | 125.7 | 125.7 | 64.5 | 241.5 |
| `val_candidate_wrecall` (kept) | 0.8845 | — | 0.8837 | 0.8832 | — | — |
| `val_pool_candidate_wrecall` | — | — | 0.9029 | 0.9125 | — | — |
| `val_wrecall` | 0.8694 | 0.8691 | 0.8687 | 0.8682 | 0.8233 | 0.8911 / 0.8912 |
| min/epoch (`_runtime` / 128) | 0.94 | **0.52** | 1.00 | 1.13 | 0.37 | 0.86 / 0.85 |

1. **The mechanism fails both acceptance criteria.** `f2` sits on the control, but it is not
   faster (1.00 min/epoch, confounded by the coverage metrics it runs). It also loses to `f1` on
   cosine (0.7270 against 0.7292) and on kept coverage (0.8837 against 0.8845). Release-score
   ranking recovers none of the extra coverage its pools hold (0.903 and 0.913), which is what
   the early read showed at epoch 27. A larger pool (`f3`) is slightly worse and 20% slower.
2. **The August U128 penalty does not reproduce.** Halving the set to about 126 fragments costs
   0.022 of `val_wrecall` and no cosine. The August sweep in the first established fact above ran
   on August code and the pre-2026-09-03 target, and its cosine penalty does not transfer to this
   base.
3. **U128 is 1.65x faster per epoch at the same cosine.** `f5` runs the control's monitoring and
   takes 0.52 min/epoch against 0.85–0.86. The coverage metrics are what make `f1`–`f3` slow:
   `f1` is the same selection at 0.94. `f4` shows where the budget starts to bind: U64 is 2.3x
   faster and loses 0.009.
4. **A second same-seed pair.** `f1` and `f5` differ only in validation-diagnostic keys and land
   0.0015 apart on the tail mean. That is wider than the 0.0004 floor from the one duplicate
   pair in the [screen read-out §3.1](../../findings/masshatter/masshatter_architecture_arms_epoch128_2026_09_22.md).
   Keep the ±0.005 gate.

Read-only W&B query, no GPU used. The U128 speed result is n=1 per arm, from wall-clock on
shared MIG hosts.

## Current state

Implemented on `research/masshatter-ion-h-selection` (uncommitted working tree, 2026-09-23):

- [mass_shatter_model.py](../../../src/fragnnet/model/mass_shatter_model.py):
  `_fragment_release_log_probabilities`, `_top_k_fragment_keep_mask`, the constructor checks, and
  the marginal branch at the existing fragment keep mask, which emits the whole-pool
  `proposal_*` and `pool_candidate_*` outputs.
- [mass_shatter_pl.py](../../../src/fragnnet/pl_model/mass_shatter_pl.py): hyperparameter wiring,
  `val_pool_candidate_wrecall`, and the `load_from_checkpoint` legacy backfill;
  [pl_utils.py](../../../src/fragnnet/utils/pl_utils.py) backfills `get_pl_hparams`.
- `config/template.yml`: the two keys, defaulting to the old behaviour.
- The three arm configs and the generate-only Fir script named in [§ Experiment](#experiment).

All five arms ran on 2026-09-23; see the tail read above. The CPU and CUDA candidate builders
are untouched.

Open points:

- Molecules whose forced set (precursor + single cuts) already reaches K keep only the forced
  set, exactly as mask order does. How often that happens at K128 on merged NIST20 is not
  measured; large acyclic molecules are the case.
- In marginal mode the charge head gets no proposal gradient, because the release score has no
  charge term. It only affected membership through the proposal before.
- If `f2` falls short of `U256` while `val_pool_candidate_wrecall` stays high, the ranking is the
  limit: a target-informed greedy top-K on the same pool would separate a weak ranker from an
  absent one before any learned fragment scorer is built.

## Validation

Run 2026-09-23 with the `FRAGNNET-GPU` interpreter:

- `tests/test_mass_shatter_model.py` — 130 passed, including the new tests:
  - the release score equals a brute-force sum over all cut masks of the intact-release event,
    equals the atom-set marginal for every acyclic fragment, and is strictly below it for every
    ring-containing one (`CC(C)CO`, `C1CC1CO`);
  - the keep mask keeps forced fragments first, breaks ties by pool order, never lets `-inf` win
    a slot, and keeps only the forced set when it fills the budget;
  - end to end on `CCCCO`/`CCCO`: pool 15/10, kept 11/10, pool proposal mass 1 per molecule;
  - **blind spot**: with the same nine kept fragments, a peak explained only by an excluded
    fragment gives a positive proposal loss and bond-head gradient in marginal mode and exactly
    zero in mask order;
  - every invalid configuration raises.
- `tests/test_mass_shatter_pl.py` — checkpoint migration through both load paths, pool-metric
  registration, and an exact key diff of the three arm configs against `c4a_base_s1`.
- Full suite at first implementation: 4,265 passed, 1 skipped, 2 xfailed.
- All three arm configs load through `runner.load_config` and build the full-size model.
- Two-epoch smoke runs of `config/debug/debug_tiny_mass_shatter.yml` in marginal mode (pool 24,
  K 12) completed on the CPU backend and on GPU with the CUDA candidate backend; both logged
  `val_pool_candidate_wrecall` above `val_candidate_wrecall` (0.618 vs 0.567 on GPU).
- **Fixed after review (2026-09-23):** with `mass_shatter_channel_support_metrics` or
  `mass_shatter_channel_diagnostics` on, `_channel_support_metrics` weighted the kept fragments by
  `proposal_fragment_logprobs`, which marginal mode makes pool-sized, and raised
  `IndexError` (mask 18 vs tensor 28 in the reproduction). The metric now reads a dedicated
  kept-aligned output, `ion_h_fragment_proposal_logprobs` (the base proposal probability of each
  scored fragment, identical to the old weights in mask order), and checks its shape. The same
  change removes a pre-existing `KeyError` when the support metrics ran outside selection-only
  mode, where no `proposal_*` tensors exist. Regression test:
  `test_channel_support_metrics_run_through_the_loss_in_every_proposal_mode` (mask order,
  marginal, joint) through the vectorized CE loss path.
- **Fixed after the combination audit (2026-09-23):** the pre-existing "never discard the last
  realizable fragment" fallback re-admitted *every* fragment tied at the per-molecule maximum.
  Release scores tie exactly on symmetric substructures (six bit-identical scores on benzene), so
  the fallback silently overrode K: benzene kept 7 fragments for every K from 1 to 7. Marginal
  mode now uses `_missing_valid_fallback_mask`, which adds one fragment (first in pool order),
  and only for a molecule whose kept set has no chemically valid fragment; mask order keeps the
  original rule unchanged. Regression tests: `test_marginal_selection_budget_holds_when_release_scores_tie`
  (kept count equals K for K = 1..10 on benzene) and
  `test_missing_valid_fallback_adds_one_fragment_only_where_none_is_kept`.
- Combination audit: five independent reviewers (loss/metric flags, model-forward flag
  combinations including CUDA and bf16, inference/export/checkpoint paths, the channel-metric
  fix, selection edge cases), every claim reproduced by running code and re-run by two refuters.
  Only the tie finding survived; the other lenses found no crash or misalignment.
- Full suite after both fixes: 4,271 passed, 1 skipped, 2 xfailed. A two-epoch training smoke
  run in marginal mode with `mass_shatter_channel_diagnostics` on completes and logs
  `fragment_weighted_precision`.
- Not a result of this change, noted for whoever reads the debug logs: the tiny smoke run reports
  per-cut-bucket coverage above 1 (`cut_2` 1.78 marginal, `cut_3` 1.52 in mask order), in both
  modes; full-size runs report bucket values ≤ 0.42.
