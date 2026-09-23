# MASShatter proposal loss — consolidated audit

> **Status:** FINDING — synthesis, read this before its four parts &nbsp;·&nbsp;
> **Updated:** 2026-09-22
>
> Everything the repository knows about `mass_shatter_proposal_loss_weight` and the term it
> weights, previously spread across four documents that each met it from a different angle.
> Each claim names its source; none is carried over without a check against the working tree.
>
> **Three results are new here.** (1) The 2026-09-13 blind-spot counterexample, re-run against
> the current function, is sharper than recorded: the loss is **exactly proportional to the
> covered intensity fraction**, so moving 60% of a spectrum's intensity onto peaks the candidate
> set cannot reach cuts both loss and gradient by 60%, and a molecule whose candidates reach
> **nothing** is a zero-loss, zero-gradient training example. (2) **The term has no one-key
> ablation.** `_setup_loss_fn` locks `proposal_loss_weight > 0` to `proposal_selection_only`,
> which also rewires the spectrum output, so no configuration in this repository can measure what
> the proposal loss is worth. (3) The diagnostic the 2026-09-13 review asked to be logged beside
> it, `candidate_wrecall`, was **switched off by default on 2026-09-18** (`17d95fb3`) and is off
> in all 86 `ma_mi_merged` configs.
>
> Source line numbers are read off a **dirty tree** (the `candidate_features` refactor is in
> flight); they drift, the symbols do not.

## 1. The term as it runs today

[`_proposal_coverage_loss`](../../../src/fragnnet/pl_model/mass_shatter_pl.py#L298), added to the
objective at [`:1769-1776`](../../../src/fragnnet/pl_model/mass_shatter_pl.py#L1769-L1776):

```
L_proposal = - Σ_{covered peaks p}  w_p · log Σ_{distinct fragments f covering p} P(f)
```

with five properties that each matter downstream:

| property | code | consequence |
|---|---|---|
| a peak and a fragment "match" when their **m/z** fall inside `inst_type_loss_tol` | `matched_peak_candidate_pairs` | fragment identity is never read (§5) |
| each unique fragment counts **once per peak**, however many ion/H channels matched | [`:372`](../../../src/fragnnet/pl_model/mass_shatter_pl.py#L372) | the term is H-independent by design |
| `w_p` is the **observed intensity** (`true_logprobs.exp()`), unless `proposal_barcode_target` | [`:401-404`](../../../src/fragnnet/pl_model/mass_shatter_pl.py#L401-L404) | barcode is on in 4/224 configs, 0/86 `ma_mi_merged` |
| **uncovered peaks contribute zero**, not a penalty | [`:395`](../../../src/fragnnet/pl_model/mass_shatter_pl.py#L395), [`:406-410`](../../../src/fragnnet/pl_model/mass_shatter_pl.py#L406-L410) | the blind spot (§3) |
| the result is tied back to `fragment_logprobs` with weight 0 so it stays differentiable | [`:413`](../../../src/fragnnet/pl_model/mass_shatter_pl.py#L413) | a fully-uncovered batch backpropagates exact zeros |

Live config census, 224 files under `config/massshatter/**` resolved against
`config/template.yml` (template defaults: weight `0.0`, `selection_only False`):

| key | whole tree | `ma_mi_merged` (86) |
|---|---|---|
| `mass_shatter_proposal_loss_weight` | `0.1` in 178, `0.0` in 46 — **no other value exists** | `0.1` in 86 |
| `mass_shatter_proposal_selection_only` | `True` in 178 | `True` in 86 |
| `mass_shatter_proposal_anchored` | `True` in 186 | `True` in 86 |
| `mass_shatter_separate_proposal_encoder` | `True` in 28 | `False` in 86 |
| `mass_shatter_proposal_only_until_epoch` | set in 38 | set in 5 |
| `mass_shatter_proposal_num_mixtures` | `1` in 216, `2` in 6, `3` in 2 | `1` in 86 |
| `mass_shatter_proposal_barcode_target` | `True` in 4 | `False` in 86 |
| `mass_shatter_candidate_coverage_metrics` | `True` in 5 (all FastGNN arms) | `False` in 86 |

The `2`/`3` mixture counts correct the head-input audit's "`proposal_num_mixtures: 1` in
223/223"; the eight exceptions are all older `merged/` arms, so its conclusion about the
current lineage stands.

## 2. Status of every claim

| claim | verdict | source |
|---|---|---|
| Uncovered target peaks contribute nothing to the loss | **CONFIRMED**, re-run today (§3) | [loss setup review](../loss_setup_review_2026_09_13.md#auxiliary-weights-proposal-coverage-and-warmup-need-separate-interpretation) |
| Proposal loss measures candidate coverage | **FALSE.** Zero coverage attains the global minimum | same |
| Proposal supervision trains the encoder | **FALSE.** Detached in 140/224 configs, 81/86 `ma_mi_merged` | [head-input asymmetry §3](masshatter_head_input_asymmetry_2026_09_22.md) |
| Connecting that gradient would help | **EVIDENCE AGAINST**: −0.011@100, −0.013@400, same seed, one key, 630 ep | [head-input asymmetry §7.1](masshatter_head_input_asymmetry_2026_09_22.md) |
| `proposal_only_until_epoch` only stages training | **FALSE.** With a shared encoder it also un-detaches the branch permanently; **38 configs confounded** | same |
| The loss reads which fragment a candidate is | **NO** — m/z only, and 2.75 connected fragments share each formula | [identity supervision](masshatter_identity_supervision_2026_09_22.md) |
| …so MASShatter needs a structural teacher | **RETRACTED** by its own source | same |
| A 0.1 coefficient means a 10% gradient contribution | **UNMEASURED.** Named as the required check in 2026-09-13, never run (§6) | [loss setup review](../loss_setup_review_2026_09_13.md) |
| The per-spectrum proposal loop costs current runs | **NO.** Batched 2026-09-18; the surviving loop is the shared-eV variant and it is off everywhere | [loss audit §5](masshatter_loss_audit_2026_09_19.md) |
| `_shared_group_proposal_coverage_loss` / `_cross_ev_fragment_delta_loss` are correct | **UNAUDITED.** Gated off by default, so only their cost was checked | same |
| U256 truncation is the practical cause of missing coverage | **WITHDRAWN** by its own source — U256 is reached in 5-13% of molecules, and E1024→E4096 buys 43-46 fragments for **0.0066 wrecall** | [proposal review](masshatter_proposal_review_2026_09_13.md#live-wb-follow-up-u256-is-usually-not-reached) |
| Selection-only is the right proposal baseline | **SUPPORTED**, one same-seed pair: +1.54 pt coverage, +0.0126 cosine at epoch 142 | [proposal review](masshatter_proposal_review_2026_09_13.md) |
| The term can be ablated to measure its worth | **NO — new (§4).** The validator couples it to two other keys and to the output wiring | this doc |

## 3. The blind spot, re-measured

`_proposal_coverage_loss` sums `-w_p · log(coverage_p)` over **covered** peaks only, and
`coverage_p ≤ 1`, so `L ≥ 0` with equality when every covered peak is covered with probability 1
**or when no peak is covered at all**. Re-run against the live function
(`scripts/debug/debug_proposal_coverage_blind_spot.py`, two peaks at 100/200 Da, 0.01 Da
tolerance):

| case | candidates | loss | Σ\|∂L/∂logits\| |
|---|---|---:|---:|
| A full coverage, P(f)=0.5 each | 100, 200 | **0.693147** | 1.000 |
| B half coverage, all probability on the hit | 100 | 0.000000 | 0.500 |
| **C zero coverage** | 300 | **0.000000** | **0.000** |
| E one fragment covers both peaks, P=1 | 100, 200 (same fragment) | 0.000000 | 1.000 |
| F case A + 20% of intensity on an unreachable peak | 100, 200 | 0.554518 | 0.800 |
| G case A + 60% of intensity on an unreachable peak | 100, 200 | 0.277259 | 0.400 |

Two readings, the second of which is new:

1. **Case C is the global minimum.** A molecule whose candidate set reaches none of its peaks is
   indistinguishable from a perfectly-covered one by loss value, and its gradient is exactly
   zero. The proposal head is never told to go look elsewhere.
2. **The loss is exactly proportional to the covered intensity fraction** (A → F → G is
   ×1.0 → ×0.8 → ×0.4, matching the covered mass). Per-spectrum, the objective therefore
   **down-weights exactly the molecules whose coverage is worst**. Whatever gradient signal a
   hard molecule could contribute is attenuated in proportion to how hard it is.

This is the objective-level statement. Its **incidence on real data is still unmeasured** — see
§6 — and the [unexplained-peak split](unexplained_peak_split_2026_09_22.md) puts the reachable
part of the residual at 7.98 of 10.89 points of validation intensity, which bounds how much of
this is fixable at all.

## 4. There is no one-key ablation of the proposal loss (new)

`_setup_loss_fn` ([`:1375-1409`](../../../src/fragnnet/pl_model/mass_shatter_pl.py#L1375-L1409))
enforces a chain that makes the weight non-independent:

- `proposal_selection_only` **requires** `proposal_loss_weight > 0`;
- `proposal_loss_weight != 0` **requires** `proposal_selection_only`;
- `proposal_selection_only` **requires** `base_ce_weight == 0`;
- `proposal_anchored` **requires** a positive proposal supervision loss (or base CE).

So turning the term off in a `ma_mi_merged` arm is a **three-key diff** — weight `0.0`,
`selection_only False`, `anchored False` — and `proposal_selection_only` is not a training-time
switch: it decides the spectrum itself,
`corrected_fragment_logits = fragment_residuals` versus
`fragment_log_probabilities + fragment_residuals`
([`mass_shatter_model.py:2681-2685`](../../../src/fragnnet/model/mass_shatter_model.py#L2681-L2685));
and because the same chain forces `proposal_anchored: False`, the base-probability column also
enters the candidate token (`use_base_probability_feature = not proposal_anchored`,
[`mass_shatter_pl.py:1010`](../../../src/fragnnet/pl_model/mass_shatter_pl.py#L1010) — the
constructor rejects that column *with* selection-only, so the two cannot be separated,
[`mass_shatter_model.py:1082-1086`](../../../src/fragnnet/model/mass_shatter_model.py#L1082-L1086)).

**Consequence.** Every "proposal loss" comparison this repository can run is a comparison of two
different output architectures. The 178 configs that enable the term all use weight exactly
`0.1`, and no config uses any other positive value — so the coefficient has never been varied
either. Nothing on record establishes that the term helps, hurts, or does nothing; the closest
evidence is the selection-only pair in [the proposal review](masshatter_proposal_review_2026_09_13.md#selection-only-improves-coverage-in-the-closest-historical-pair),
which changes the weight, the wiring and the base-CE weight together.

This is the same shape as the [regularization audit](masshatter_regularization_audit_2026_09_19.md)'s
finding for `head_dropout`/`wd`/`lr`: a hyperparameter that is identical in every run is not a
tuned value, it is an untested one.

## 5. What the term trains, and what it never sees

**It trains two head MLPs.** `detach_proposal_inputs = proposal_selection_only and not
proposal_input_gradients_enabled`
([`mass_shatter_model.py:2181-2183`](../../../src/fragnnet/model/mass_shatter_model.py#L2181-L2183)),
and `proposal_input_gradients_enabled` starts `False` and is written only by
`_apply_staged_training`, which returns early for a null warmup
([`mass_shatter_pl.py:1127-1138`](../../../src/fragnnet/pl_model/mass_shatter_pl.py#L1127-L1138)):

| steady-state routing | configs | `ma_mi_merged` |
|---|---:|---:|
| proposal inputs **detached for the whole run** | 140 | 81 |
| staging set + shared encoder → gradients on in **every** epoch | 38 | 5 |
| `proposal_selection_only` off → no detach | 46 | 0 |

So `proposal_loss_weight: 0.1` updates `bond_break_head` and `charge_head` and **no encoder or
context parameter**. The one run that connects it (`jvaj1lyw` vs `tkkesus9`, same seed, one key)
is 0.011-0.013 cosine **behind** across epochs 100-400 — so the detachment is not a bug to fix.
Full table and the confound argument in
[head-input asymmetry §7.1](masshatter_head_input_asymmetry_2026_09_22.md#71-the-obvious-fix--connect-the-proposal-gradient--already-ran-and-lost).

**It never sees fragment identity.** Matching is m/z-only, and 2.75 distinct connected fragments
share each formula (p50 2.56, p90 3.97, in 82.3% of molecules) — a degeneracy no formula label
resolves. The [identity-supervision finding](masshatter_identity_supervision_2026_09_22.md)
measures this and then **retracts** the teacher recommendation it drew from it: same-formula
isomers sum at one m/z, so choosing between them changes the predicted spectrum by nothing.

**It never sees CE, in the arms that share proposals.** `share_proposal_across_ev` builds the
proposal context with `include_collision_energy=False` — on in 2/224 configs, and the one head
that escapes it is `ion_mode_proposal_head`, off in 224/224
([head-input asymmetry §4.7](masshatter_head_input_asymmetry_2026_09_22.md)).

## 6. What is still not audited

Ordered by how much the answer would change what we do.

1. **What the term is worth.** No ablation is expressible (§4). Requires either a config-shape
   change or accepting a three-key comparison and saying so.
2. **Per-term gradient norms.** "A 0.1 coefficient does not establish a 10% gradient
   contribution" was the 2026-09-13 review's explicit acceptance check and has never been run.
   §5 partly moots it for the current lineage — the norm on every shared parameter is exactly
   zero — but not for the two head MLPs it does reach.
3. **Blind-spot incidence on real data.** The counterexample is algebraic; how much training
   intensity actually falls in case C or is attenuated as in case G is unmeasured, and
   **`candidate_wrecall` — the metric that would show it — has been off by default since
   `17d95fb3` (2026-09-18) and is off in all 86 `ma_mi_merged` configs**, gated at
   [`mass_shatter_pl.py:2179`](../../../src/fragnnet/pl_model/mass_shatter_pl.py#L2179). Before
   that commit it was computed unconditionally, which is where the older `val_candidate_wrecall`
   figures quoted in the proposal review come from. **Note the name collision:**
   `candidate_wrecall` (what the *candidate set* can reach) is not the spectrum metric `wrecall`
   (what the *prediction* matches) used by the
   [overfit read-out](masshatter_overfit_readout_2026_09_22.md); the "coverage already transfers"
   argument rests on the latter.
4. **The two gated variants.** `_shared_group_proposal_coverage_loss`
   ([`:563`](../../../src/fragnnet/pl_model/mass_shatter_pl.py#L563), gated by
   `share_proposal_across_ev`) and `_cross_ev_fragment_delta_loss` have had a cost and gating
   check only; their semantics have never been reviewed, and both are off in every config.
5. **`proposal_barcode_target`.** Four configs, all pre-`tolerance_consensus`, so their results
   are not comparable to the current band.

## 7. What to do

1. **Split the two behaviours of `proposal_only_until_epoch`** (~10 lines). It is what makes the
   one gradient-connection measurement ambiguous and what confounds 38 historical runs. Value is
   disconfounding, not cosine. (Carried over unchanged from
   [head-input asymmetry §7.2](masshatter_head_input_asymmetry_2026_09_22.md).)
2. **Turn `candidate_coverage_metrics` back on for one arm** and read `candidate_wrecall` beside
   the proposal loss for an epoch. One match-mask pass per spectrum; it answers §6.3 without a
   new run if folded into an arm that is going to run anyway.
3. **Decide whether the no-ablation state is acceptable.** If the term stays unmeasured, say so
   in the template comment; if it should be measurable, the minimal change is to allow
   `proposal_selection_only` with `proposal_loss_weight: 0.0`. The validator forbids that today
   and records no rationale at the check. What it would produce is an **entirely unsupervised
   proposal branch** that still selects candidates from its initialization — which is exactly
   the ablation, and cheap: one key, one arm, everything else held.
4. **Nothing on the coverage axis.** U256 truncation is withdrawn as the practical cause, the
   E1024→E4096 gain is 0.0066 wrecall, the connected-fragment beam has an architectural ceiling,
   and the [overfit read-out](masshatter_overfit_readout_2026_09_22.md) puts ~98% of the
   train/validation gap in intensity allocation rather than in anything the proposal branch
   controls. The proposal loss is not where the headline metric is.

## 8. Sources and provenance

Consolidates, and does not replace:

- [loss_setup_review_2026_09_13.md](../loss_setup_review_2026_09_13.md) — canonical for the
  blind-spot counterexample, the gradient-routing warning, and the auxiliary-weight caveats.
  Also covers FraGNNet defects outside this doc's scope.
- [masshatter_proposal_review_2026_09_13.md](masshatter_proposal_review_2026_09_13.md) — the
  proposal *branch* review: candidate budgets, deterministic selection, diversity, mixtures, and
  the live W&B follow-ups that withdrew the U256 hypothesis.
- [masshatter_loss_audit_2026_09_19.md](masshatter_loss_audit_2026_09_19.md) — implementation
  audit of the loss path after the batching refactor; the proposal term appears there as a cost
  and gating question.
- [masshatter_head_input_asymmetry_2026_09_22.md](masshatter_head_input_asymmetry_2026_09_22.md)
  — what the proposal heads read, the detachment census, and the one gradient-connection
  measurement.
- [masshatter_identity_supervision_2026_09_22.md](masshatter_identity_supervision_2026_09_22.md)
  — what the objective can and cannot see, and the retracted teacher recommendation.

New work in this document: the live 224-config census (§1), the re-run counterexample and its
proportionality result (§3, `scripts/debug/debug_proposal_coverage_blind_spot.py`), the
no-ablation finding (§4), and the `candidate_wrecall` regression (§6.3). Documentation and one
debug script only — no model code, configuration, dataset or training state changed, and no run
was launched.
