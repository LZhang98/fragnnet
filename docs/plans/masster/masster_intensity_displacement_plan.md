# Intensity displacement: where MASSter puts the intensity it misplaces

> **Status:** ACTIVE · **Updated:** 2026-09-24
> **Phase 0 is done** ([finding](../../findings/masster/masster_intensity_displacement_phase0_2026_09_24.md)).
>
> - **G0 condition 1 passes.** 67% (val) and 63% (test) of the displacement is reducible error
>   that the two runs share.
> - **G0 condition 2 fails against the proper null.** Against a null that keeps mass on the real
>   peaks, no class exceeds 1.08x. Displacement is systematic misordering among real peaks, not
>   neighbour confusion.
> - **The one structured part is the precursor/parent fallback below NCE 60.** There a
>   formula-only learner stacked with the model adds +0.03 pairwise AUC on the
>   precursor-versus-contender decision.
>
> The class-targeted items (1.2, 2C, the H2/H3 split) are dropped.
>
> **Phase 1.4b is done: gate G1 is not passed**
> ([finding](../../findings/masster/masster_bond_local_learner_2026_09_24.md)).
>
> - **An identity-blind bond-local learner adds a little.** Stacked with the model at peak level,
>   it gives +0.0022 (val) and +0.0025 (test) group cosine over a fitted temperature, SE 0.0003.
>   Beyond per-NCE temperature and precursor offsets, it gives +0.0017 and +0.0028.
> - **The gain sits where transfer fails.** It is largest on novel molecules and at NCE ≤ 40,
>   and near zero above NCE 60.
> - **Alone, the learner orders bonds worse than the model**, even on novel molecules.
> - **Verdict:** between FAIL (+0.002) and PASS (+0.005). 2A is not submitted.
>
> One cheap offline retry is open: a learner trained on the peak-level objective with hashed
> environments. Phase 2 is not started. Scope:
> unmerged `[M+H]+`/FT on `data/split/nist20_inchikey`. The memorization/overfitting line is owned
> by the [ranking-transfer plan](masster_ranking_transfer_plan.md); the boundary is set out in
> §1.

## 1. Question and boundary

**Intensity displacement** is validation target intensity that sits on a candidate the model
already proposes, but that the model assigns to a *different* candidate. By the
[ranking decomposition](../../findings/masster/masster_ranking_decomposition_2026_09_23.md),
that is ranking (0.121) plus calibration (0.104 ceiling) out of a 0.238 val gap, against
coverage 0.014.

Both terms are near zero on training molecules (ranking 0.011, calibration 0.019), so every
intervention on displacement is also a generalization intervention. The two plans split by
mechanism, not by metric:

| question | owner |
| --- | --- |
| Why the train fit does not transfer: lookup capacity, optimizer noise, the boundary block, EMA, allocator dropout/width, more molecules, distillation | [ranking-transfer plan](masster_ranking_transfer_plan.md) |
| Where the misplaced mass goes; how much of it is **shared by independently trained runs** (bias, not variance); whether an identity-blind local rule predicts it; calibration shape; the low-CE precursor fallback | this plan |

Rules of engagement:

- Do not vary keys the other plan varies: `masster_intensity_transformer_dropout`,
  `masster_intensity_transformer_dim`, `ema_decay`, `masster_candidate_boundary_dropout`,
  `masster_candidate_atom_dropout`, `boundary_bond_encoder`.
- Share its control `3bej6fq7` (tail mean 0.7298 over epochs 54-63) and its exports. If one of
  its arms passes, rebase Phase 2 arms on the winner instead of on f0.
- The ranking decomposition's third lever, "information that generalizes across molecules, such
  as bond-lability priors not learned from this data," has no owner. This plan takes it.

## 2. Established facts

Numbers are f1 (`6h67zps9`) on its cleaned val target unless stated; test agrees within 0.01.

**Size and shape of the misplacement**
([ranking decomposition](../../findings/masster/masster_ranking_decomposition_2026_09_23.md)):

- The base peak is missed in 43% of spectra, and 97% of those misses are covered.
- When missed, the true base peak is the model's #2 in 37% of cases and in its top 3 in 52%.
  72% of misses land within 60 Da of the true base peak, but only 19% are exactly one common
  neutral loss away.
- Ranking rises from 0.065 (NCE ≤ 20) to 0.151 (NCE > 60). Calibration is flat at 0.095-0.108.
- The model is too sharp: its entropy is 0.15 nats below the target's. A global `P^0.75` adds
  +0.007 on test. The per-spectrum best γ adds +0.039, and the isotonic ceiling is +0.104.
- **Most of the ranking error is shared between runs.** The f0+f1 ensemble cuts ranking only from
  0.120 to 0.110. f0 and f1 share a seed and differ in target cleaning, so this understates the
  between-run variance.

**Precursor fallback.** At NCE ≤ 20 the base-peak hit rate is 0.88 when the precursor is the base
peak and 0.42 when it is not. In 69% of the second group's misses the model predicts the
precursor. On train the same hit rate is 0.90, so the fallback appears only on unseen molecules
([overfit read-out §7](../../findings/masster/masster_overfit_readout_2026_09_22.md#7-intensity-memorization-threat-model)).
On FraGNNet a **global** precursor penalty was swept and rejected (`b = 0` selected). The doc
asks for a structure-dependent precursor-versus-contender test, which nobody has run
([dominant-peak decomposition](../../findings/fragnnet/dominant_peak_outcome_decomposition.md)).

**What the misses look like on FraGNNet** (same doc, `pqx7ywc9` test): 66% land within ±60 Da on
discrete neutral-loss offsets. The precursor is beaten by one facile, structure-specific cleavage
(H₂O 7.1%, CO₂ 5.6%, NH₃ 3.3%, C₆H₁₀O₅ 2.6%, median loss 97 Da). That doc's reading is
*labile-bond identity*: the model knows the average molecule's behaviour, not this molecule's
weakest bond.

What will not carry the fix:

- Formula statistics: a formula-only learner blended in adds +0.005
  ([formula-only baseline](../../findings/masster/masster_formula_only_baseline_2026_09_23.md)).
- A molecule-independent per-loss reweighting closes 9.3% of the cross-adduct gap; 90.7% is
  molecule-specific ([cross-adduct headroom](../../findings/cross_adduct_loss_headroom_2026_09_17.md)).
- Objective swaps: JSS, KL, Hellinger and direct cosine all land at or below CE, and a train-set
  ranking loss has little to fit, since train ranking is 0.011.
- A global temperature, beyond the free +0.007.

**Local environment carries the signal, sparsely.** In the
[FluxMPNN post-mortem](../fragnnet/fluxmpnn_plan.md), bond environment explains branching at
ICC 0.72 at radius 2, saturating at radius 3. Only 5.8% of edges sit in a radius-2 bucket with
≥20 support. So a local rule is informative where the environment is well supported, and data
is the only fix where it is not.

**Code: the final allocation has no identity-blind path.**

- With `masster_proposal_selection_only: true` (every production config), the final
  fragment distribution is `log_softmax` over the intensity transformer's residuals alone
  ([`masster_model.py:2983-2994`](../../../src/fragnnet/model/masster_model.py#L2983-L2994)).
  Ion/H is a per-fragment softmax added on top (`:3280-3282`). The bond-local proposal only
  selects the pool.
- The encoder is `GINE_GSA`, which adds exact per-graph global self-attention
  ([`nn_blocks.py:9`](../../../src/fragnnet/model/nn_blocks.py#L9)). Every atom embedding, and
  so every head input, is globally contextualized. This is the route by which "identity reaches
  every head input."
- Selection-only was adopted over the anchored `l_i + r_i` form on one same-seed 2026-08 pair
  (`p0fk89de` vs `vu5nwu9q`, +0.01261 cosine at epoch 142). That pair's joint arm also carried
  `masster_base_ce_weight: 0.1`, which the 0.1-0.3 sweep showed to be harmful through a
  shared-head gradient ([MASSter plan](masster_plan.md), "fixed base-only CE trades final
  cosine for support"). A structural prior in the final score has therefore never had a clean
  test, and never one from an identity-blind encoder.

## 3. Hypotheses and the lever each implies

| id | displacement class | signature in the transport analysis | lever |
| --- | --- | --- | --- |
| H1 | precursor ↔ first fragment | excess on the precursor, deficit on one fragment, low NCE | molecule-specific lability (2A) |
| H2 | nested, one loss (parent ↔ child on one path) | donor and recipient formulas nested and one neutral-loss step apart | molecule-specific lability of that loss (2A) |
| H3 | non-nested (sibling or other branch) | formulas not nested | which bond breaks: comparative, local (2A) |
| H4 | H shift | same heavy-atom formula, different H | ion/H channel head (2C) |
| H5 | calibration shape | order right, heights wrong; per-spectrum sharpness varies | conditional sharpness, post hoc (1.5) |
| H6 | rare environment | displacement concentrated where the local environment has little train support | data: the other plan's NIST23/distillation line |

H1-H3 share a mechanism. The model has no identity-blind estimate of which bond in *this*
molecule is labile. H6 is the null for all three: the same displacement driven by environments
the data never pins down.

**Phase 0 outcome (2026-09-24):**

- **H1 is supported.** Below NCE 60 the model over-holds the heavier member of nested pairs: at
  NCE ≤ 20, 0.64 of that mass sits on the heavier member against 0.19 for the null. Formula-level
  loss identity is complementary to the model on that decision.
- **H2 and H3 cannot be told apart from a random reordering of the real peaks** (0.90-1.08x).
- **H4 is dropped:** 6% of the mass, 0.96x the null.
- **H5 is untested** (1.5). **H6 is open** (1.3).

## 4. Phase 0: offline on existing exports (no training, no new export)

Inputs, all local: the cross-eval exports in `output/cross_eval_mh_ft_clean_2026_09_23/cross_eval/`
(f0/f1 × raw/filtered × val/test, plus the 25.6k `filtered_train` sample). Also the
nearest-subformula assignment in `analysis/formula_baseline/`, the novelty table in
`analysis/variance_novelty/`, and `ranking_decomp.py`'s candidate space (merged exact m/z, target
mapped to the nearest candidate within 10 ppm / 200 Da floor). New scripts belong in
`scripts/eval/`, so the findings are reproducible from Git.

**0.1 Bias/variance split of the error.** Per spectrum, in `ranking_decomp.py`'s L2-normalized
candidate space, take the two runs' error vectors `e0 = P̂0 − T̂` and `e1 = P̂1 − T̂`. With
`ē = (e0 + e1)/2` and `d = (e0 − e1)/2`, the shared (bias) part is `|ē|² − |d|²` and the
run-specific part is `2|d|²`. Report the shared share overall, by NCE and by novelty bin. Repeat
on f1 against its two-checkpoint weight average to bound optimizer noise.

- Caveat: f0/f1 share a seed, so this is an upper bound on the shared share. The other plan's
  pending seed pair tightens it.
- **Replicate floor:** the same statistic between replicate targets of one molecule at the same
  NCE (val molecules with repeated spectra). This is the irreducible part of "shared" error.

**0.2 Transport anatomy.** Per spectrum, excess `(P̂ − T̂)+` and deficit `(T̂ − P̂)+` on the covered
support, L1-normalized. Solve exact optimal transport on the nonzero entries (tens per spectrum,
`scipy.optimize.linprog`), with a ground cost ordered by the §3 classes:

- precursor;
- H-shift (same heavy atoms);
- nested at one step of a neutral-loss vocabulary seeded from the dominant-peak list;
- nested at two or more steps;
- non-nested;
- ambiguous assignment (a second subformula within 10 ppm, about 10% of peaks), reported apart.

Report the share of displaced mass per class, for the shared component `ē` and for single runs,
by NCE and novelty, on val and on the train sample. For nested pairs, also report the direction:
mass stuck on the heavier formula or the lighter one.

Two controls, both required before reading a share:

- **Chance:** within-spectrum permutations of `P̂` over the covered support give the class shares
  a structureless displacement would have. The FraGNNet decomposition learned this the hard way
  (47% chance, not 50%).
- **Cost robustness:** reorder the class costs and check that the leading class does not change.

**0.3 Precursor versus contender (H1), model-free.** At NCE ≤ 30, pair the precursor with the
strongest covered non-precursor contender and read the observed soft log-ratio. Keep both
precursor-win and fragment-win examples. This is the test the dominant-peak doc prescribes. Its
features and learner are the ones in Phase 1.4, restricted to formula-level inputs here.

**Gate G0.** Continue to Phase 1 only if both hold:

- the shared share of the displacement error, minus the replicate floor, is at least half;
- one class (or H1-H3 together) carries at least 1.5x its chance share of the shared displaced
  mass.

If the error is mostly run-specific, it belongs to the ranking-transfer plan: stop here and keep
only the calibration work (1.5). If shares match chance, displacement is unstructured, and the
class-targeted arms in Phase 2 have no premise.

**G0 result (2026-09-24).**

- The first condition passes: the reducible shared share is 67.0% on val and 62.7% on test.
- The second passes only against the naive null (one-loss 1.75x).
- A support-preserving null was added after the first read. It shuffles the prediction within
  the target-supported candidates and within the rest separately. Real peaks are themselves
  linked by common losses, so this null is the right control, and against it no class exceeds
  1.08x.
- The "shares match chance" branch therefore applies. The class-targeted arms are dropped, and
  the only structured part is the fallback below NCE 60.
- 0.3's formula-level test is complementary, not a replacement for the model: accuracy 0.714
  against the model's 0.754, but stacked AUC 0.798 → 0.829 on val and 0.807 → 0.838 on test.

Tables and limits are in the
[finding](../../findings/masster/masster_intensity_displacement_phase0_2026_09_24.md).

## 5. Phase 1: one new export, still no training

**1.1 Export fragment identity.** Add an opt-in flag to `scripts/eval/run_save_inference_eval.py`
(the analogue of FraGNNet's `--return_cc_hash`). For each retained fragment it emits the atom
index list, the cut (boundary) bond indices and the final residual logit `r_i`. For each channel
row it emits the fragment index and the H shift. The model already builds these tensors for the
boundary pool and the channel grid. The default export is unchanged. Re-export f0/f1 val and the
train sample on the local card; the memorization probe took about 45 min per checkpoint at the
same scale.

**1.2 Fragment-level classes. DROPPED 2026-09-24:** the formula-level class mix already matches a
random reordering of the real peaks, so finer classes would not change the reading. Kept for the
record:

- atom-set containment for nested pairs;
- shared parent cut for siblings;
- exact H shift.

This removes the 10% assignment ambiguity for the model's own candidates. Same-formula isomers
still merge at one m/z and stay invisible, as they are to the metric.

**1.3 Environment-support stratification (H2/H3 against H6).**

- For each displaced donor/recipient pair, take the bond(s) that distinguish the two fragments.
  Hash their radius-1 and radius-2 environments, with the same construction as the FluxMPNN ICC
  audit.
- Count each hash's support: training molecules in which a bond with that environment is cut in a
  covered, target-supported fragment.
- Plot the shared displaced mass against support.

**1.4 Local-rule baseline.** This is the structural analogue of the formula-only learner.
**Re-aimed 2026-09-24:** Phase 0 found no preferred pair type beyond the precursor fallback, so
the pairs are chosen by target intensity, not by class. They are the target's top pairs of real
peaks in each spectrum. First comes the precursor-versus-contender decision at NCE ≤ 30, where a
formula-only learner already stacks for +0.03 AUC; bond-local structure must add to that. The
pair-type list below is superseded by this.

- A gradient-boosted model sees only bond-local inputs: the cut bonds' radius-0/1/2 environment
  hashes with back-off, bond order and ring membership, the lost neutral's formula,
  charge-side descriptors, and NCE. It sees no molecule-level feature and no fingerprint of the
  whole molecule.
- It is fitted on training molecules to predict the observed soft log-ratio of three pair types:
  precursor vs strongest contender, parent vs one-loss child, and sibling cuts of one parent.
- It uses responsibility weighting on soft intensities, not hard m/z-match labels.
- Read it two ways on val. First, pairwise ordering accuracy against MASSter's implied ratio
  on the shared-displacement pairs, by environment-support bin. Second, cosine after a
  log-linear correction of the exported predictions on those pairs, stacked on `P^0.75` and on
  the formula blend.

**1.4b Bond-level form (run first, 2026-09-24).** After the bond-type check, the unit is the
candidate bond, not the pair. `bond_type_ce_allocation.py --bond_records_fp` writes each
spectrum's candidate bonds with the model's mass and the target mass. Target mass is split
equally over shared m/z (model-free); the responsibility split is kept as a check.
`scripts/eval/bond_local_learner.py` fits identity-blind learners on train molecules:

- GBDT, Poisson on the bond's share;
- MLP with the model's listwise CE;
- a type-only GBDT (bond type, NCE, candidate count).

Inputs are bond order and ring context, endpoint atoms, element counts within two bonds, the
formula of the piece a non-ring cut releases, NCE, candidate count, and molecule heavy-atom
count and mass. Size is a deliberate deviation from "no molecule-level feature": it is not
identity, and it drives precursor survival. Radius-2 element counts replace environment hashes.

The read is at peak level, in the form 2A would take. Each fragment is re-scored as
`a log w_f + c mean log p_b + b mean S(b)` over its cut bonds, then renormalized. Channels keep
the model's ion/H split and are summed into m/z groups. The weights are fitted on the other
split: val and test share no molecules. The score is group cosine, with molecule-clustered
SEs. There are two controls:

- `model_T`: temperature only, which includes the free `P^0.75`;
- `stack_logp`: the model's own bond-share aggregate, with no learner.

Bond-level ordering and TV tables are kept as description only. Isomeric fragments share an
m/z, so the bond a target peak belongs to is a convention there. On train, the equal split gives
0.25 bond TV against 0.07 for the responsibility split.

Pre-registered gate. It was first written at bond level (type TV) and moved to peak level after
the train-split isomer check, before any val/test number was seen.

- PASS if `stack_all` beats both `model_T` and `stack_logp` by at least +0.005 group cosine
  on both val and test, and the gain on novel molecules (nearest-train Tanimoto < 0.5) is no
  smaller than on familiar ones.
- FAIL if the gain over `model_T` is below +0.002 on either split. A fail means the
  cross-type ordering is not recoverable from local structure, and 2A is dropped.

**1.5 Conditional sharpness (H5).**

- Split val by molecule. On one half, regress each spectrum's optimal γ on inputs available at
  inference: NCE, precursor m/z, predicted entropy, top-1/top-2 margin, candidate count, and
  nearest-train Tanimoto.
- Report the realized gain on the other half and on test, against global γ = 0.75, and stacked
  with the formula blend. The stacking has never been measured.
- A training-time temperature is not proposed. The over-sharpness is the same on train
  (−0.12 nats), and cross-entropy training would refit it.

**Gate G1 (training).** Run arm 2A only if both hold:

- the local baseline beats MASSter's pairwise accuracy on shared-displacement pairs in
  well-supported environments (radius-2 support ≥ 20);
- its correction adds at least +0.005 on test (molecule-clustered SE) on top of `P^0.75` and the
  formula blend.

If displacement concentrates in rare environments, it is H6: hand the numbers to the
ranking-transfer plan's data line and do not build 2A.

**Gate G1-inference.** If conditional γ gains at least +0.005 on test beyond global γ, add an
explicit inference option (default off; γ recorded in the export). The global `P^0.75` goes in
the same option.

## 6. Phase 2: training arms (only the branch whose gate passed)

Every arm is one mechanism off f0 (unmerged `[M+H]+`/FT, 64 epochs, 308 steps/epoch, seed 397623).
It is read on the tail mean over epochs 54-63 against `3bej6fq7` 0.7298, the same as the other
plan. `upload_checkpoints: true`. Profile `fir:h100-3g.40gb`, 12 h box.

Beyond the tail mean, each arm is read offline:

- the displacement share of the class it targets (Phase 0.2/1.2);
- val and train ranking gaps;
- the novelty bins;
- the environment-support bins.

**2A. Identity-blind local lability prior (H1-H3; needs G1).**

`log P(i) = log_softmax_i(r_i + b_i)`. `b_i` is fragment `i`'s release log-probability under
per-bond Bernoulli scores, the closed form `_fragment_release_log_probabilities` already
computes (`masster_model.py:183-245`). The bond scores come from a strictly local encoder:
k-hop GINE with no global self-attention, NCE as its only context, and its own parameters. It
differs from the 2026-08 anchored design, which lost, in three ways:

1. The prior cannot see molecule identity beyond k hops, so it cannot become a lookup.
2. It has its own base-only CE on `softmax_i(b_i)` with detached ion/H. That loss reaches only
   the local encoder, so it cannot interfere with the transformer. The old design's shared-head
   gradient did interfere, and the fix that plan recommended never ran. The base loss is what
   forces `b_i` to be a competent predictor on its own. Without it, the higher-capacity global
   path can absorb the signal and leave `b_i` near zero.
3. The transformer does not receive `b_i`, so it cannot cancel it (the `proposal_anchored`
   rule).

New keys (template-declared, default off): the prior's on/off, its GNN depth, and its base-loss
weight.

Arms:

- `l1`: k = 2 (the ICC radius), base weight 0.1;
- `l2`: k = 3, to test the saturation.

Before submission, check on the local card that `b_i` alone scores a val cosine the local-rule
baseline (1.4) predicts it should. If the in-model prior cannot match the offline baseline, the
arm tests the encoder, not the hypothesis.

Pass: the tail-mean gate, plus a lower shared displaced mass in H1-H3, concentrated in
well-supported environments.

H1 gets no loss of its own. At low CE the precursor-versus-contender decision is the weakest
bond's lability, which is what `b_i` estimates. A precursor pairwise term, like any train-set
ranking term, has almost nothing to fit.

**2C. H-shift channel. DROPPED 2026-09-24:** H-shift moves are 6% of the displaced mass, at
0.96x the support-preserving null. Kept for the record: The
[ion/H plan](masster_ion_h_channel_selection_plan.md) has the machinery: a separate support
head, `masster_ion_h_support_loss_weight`, and CE-aware support. Its Phase 2
fixed-checkpoint sweep needs a checkpoint with a separate support head, and f0/f1 have none. The
arm is therefore f0 plus the support head at the weight that plan fixes (0.1). It is read first
on H-shift displacement, then on cosine.

**Not proposed**, with reasons in §2: a train-set ranking or precursor pairwise loss, a global
precursor bias, objective swaps (JSS/KL/Hellinger/cosine, peak-list matching), escape, formula
features, and allocator-capacity or boundary arms (the other plan's). A candidate-attention
relation bias for H3 is also out: the pairwise mass bias, which already encodes neutral-loss
mass differences, was an exact null (−0.0002) at matched topology.

## 7. Acceptance

The plan succeeds if a Phase 2 arm, or an inference option, does all of the following:

1. Raises the test cosine by more than the gate: tail mean for arms, +0.005 molecule-clustered
   for inference options.
2. Lowers the shared displaced mass in the class it targets, measured with the Phase 0 scripts
   on its own export.
3. Stacks with whatever the ranking-transfer plan promotes.

Also re-run 0.1-0.2 on the MassSpecGym run (`1imawkx1` or its successor) before any MSGYM arm,
since the class shares may differ on that population.

## 8. Costs

- Phase 0: CPU only, local, about a day of scripting on existing pickles and exports.
- Phase 1: one export flag and its test, plus two local re-exports (about 45 min each on the
  8 GB card at the probe's batch settings). A CPU GBM on pair tables.
- Phase 2: one Fir job per arm, about 4-5 h of training in a 12 h box, like `h1`/`h2`. 2A needs
  new model code, a template key set, tests, and a smoke run before submission.

## 9. Open questions

- Whether f0/f1's shared error survives a pure seed pair. It is owned by the other plan and
  gates the size of everything here.
- Whether the merged `c4a_base` lineage shows the same class shares. That lineage has no
  uploaded checkpoints, so it needs `upload_checkpoints: true` on its next run.
- Whether the "72% within 60 Da, 19% one common loss" split is mostly H-shift, multi-step
  nested, or sibling. Phase 0.2 answers this first.

## Completed work

**Phase 0 (2026-09-24).**

- Code: [`scripts/eval/intensity_displacement.py`](../../../scripts/eval/intensity_displacement.py)
  has `bias`, `transport`, `precursor` and `summary` stages.
- Tests: [`tests/test_intensity_displacement.py`](../../../tests/test_intensity_displacement.py)
  (7 tests) cover formula assignment, pair classes, transport, and the exact bias/variance
  identity.
- Driver: `output/cross_eval_mh_ft_clean_2026_09_23/analysis/displacement/run_phase0.sh`
  (local), about 50 minutes on 10 CPU workers.
- The raw val/test targets were copied from a `/tmp` scratchpad into
  `output/cross_eval_mh_ft_clean_2026_09_23/targets/`.
- Results: [finding](../../findings/masster/masster_intensity_displacement_phase0_2026_09_24.md).

**Literature (2026-09-24):** [note](../../notes/intensity_displacement_literature_2026_09_24.md). It adds a per-instrument CE-offset check, a data-side source of shared bias, to run first once the raw NIST20 `Instrument` field is joined on the cluster. It also refines 2A: CE-dependent per-bond break curves (threshold, slope), with precursor survival taken from the same prior, as in CFM-ID.

**Bond-type check (2026-09-24):** [finding addendum](../../findings/masster/masster_intensity_displacement_phase0_2026_09_24.md#addendum-bond-type-check-iceberg-style-2026-09-24). Aggregate bond-type shares by CE match the target on val as on train, to about 0.01. Of the val−train bond-level gap, 59% is between bond types within a molecule, 26% within a type and 16% the precursor. 2A's prior must therefore rank bonds of different types on one scale.

**Phase 1.4b, bond-local learner (2026-09-24):** [finding](../../findings/masster/masster_bond_local_learner_2026_09_24.md).
The probe gained `--bond_records_fp` and `--fragment_level`. `scripts/eval/bond_local_learner.py`
fits identity-blind GBDT/MLP learners on 12,055 train molecules. It stacks them onto f1's
fragment scores at peak level, cross-fitted between val and test.

- **Gain.** `stack_all` − `model_T` is +0.0022 ± 0.0003 (val) and +0.0025 ± 0.0003 (test). Beyond
  per-NCE temperature and precursor offsets it is +0.0017 / +0.0028. On novel molecules it is
  +0.0032 / +0.0037.
- **What it is not.** The learner on the precursor alone adds 0.0000, and the bond-type × NCE
  learner adds +0.0006 / +0.0004.
- **Gate G1 is not passed.** The result is in the grey zone between FAIL and PASS.
- **Open.** One offline retry with a peak-level-trained learner and hashed environments. It
  needs no new inference. Build 2A only if that reaches +0.005 beyond `ctrl_nce` on both splits.

The precursor-versus-contender pair form of 1.4 is superseded. At peak level, the learner's
precursor term alone adds 0.0000 over one temperature.
