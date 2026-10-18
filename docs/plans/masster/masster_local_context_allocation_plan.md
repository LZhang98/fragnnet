# Local fragment scoring with a constrained context correction

> **Status:** STOPPED · **Updated:** 2026-10-06
> Stage 1 (frozen-token probe) is built and was run locally on three producers: the two local
> `i1` anchors and the full-data `i1` (user decision, 2026-09-29: `i1`, not `i4`, for the
> full-data token check and 3A; ordinary `i3` or `i4` is the 3B baseline). Caches reproduce every
> checkpoint's logged validation cosine. The user stopped the fits at A1 epoch 17, before any
> lock or locked read. No Stage 3 training. See [implementation and execution state](#implementation-and-execution-state-2026-09-29).

## Objective and acceptance

Test whether a local cut scorer plus a regularized global correction improves held-out
fragment allocation without the underfitting caused by removing molecular context outright.
Keep the spectrum-wide fragment softmax, conditional ion/H normalization, and mass aggregation.
Parent/child normalization, fragmentation flow, new supervision, and candidate generation
changes are outside this experiment.

Success requires improved spectrum accuracy and unfamiliar-molecule ranking against the
untouched checkpoint, independently calibrated controls, and an unrestricted matched scorer.
The primary read is calibrated production cosine. Smaller train/validation gaps,
smaller context logits, or lower train fit alone are not success. The local-only arm diagnoses
information loss; it is not the intended replacement architecture.

## Evidence and ownership

- The [memorization-route plan](masster_memorization_route_plan.md) owns the broader transfer
  work and the `i1`/`i3`/`i4` results. Its light-side restrictions already include underfitting
  examples. This plan owns only the local/context allocation experiment.
- The [base-peak finding](../../findings/masster/masster_i1_base_peak_errors_2026_09_28.md)
  motivates nested and sibling ranking metrics. It establishes a transfer deficit, not that
  whole-molecule information is unnecessary.
- The bond-local learner (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_bond_local_learner_2026_09_24.md`)
  was weaker alone and added only a small complementary gain on the older `f1` checkpoint,
  especially little at high energy. Its group cosine is not production cosine. These results
  argue against assuming that locality is sufficient.
- [Displacement arm 2A](masster_intensity_displacement_plan.md) remains the canonical older
  proposal: a Bernoulli release prior and auxiliary local loss. Here the local score is an
  additive boundary score; neither Bernoulli semantics nor an auxiliary local loss is assumed.
- The [retained-connectivity screen](../../findings/masster/masster_fragment_connectivity_local_screen_2026_09_29.md)
  was not positive. Adding a representation to an unrestricted token differs from penalizing
  the global contribution, but this distinction is a hypothesis, not evidence of a gain.

- The [memorization audit](../../findings/masster/masster_memorization_attempts_audit_2026_09_27.md)
  records `r1`, a one-layer local cut encoder, at -0.0029 versus its control (within seed
  spread). Route removal did not improve the audited runs; noise on a redundant route (`k1`)
  helped modestly. This is evidence against assuming stronger restrictions are better.
- [Post-cut calibration](../../findings/masster/masster_i1_postcut_information_gate_2026_09_29.md)
  recovers about +0.0053 on `i1`; the base-peak finding reports +0.0052 from `P^gamma`.
  Those are candidate-space diagnostic gains, not interchangeable with production cosine.
  The local learner's small gain, descriptors' near-null, and `fc1`'s calibrated +0.0008
  make calibration a primary comparator, not a follow-up explanation.

Two qualifications apply to older locality claims: a bounded neighborhood can still identify
rare environments or a small molecule, and withholding local logits from the global head does
not prevent that head from learning a correlated cancellation. Do not claim identity removal.

## Proposed score and constraint

For spectrum x and chemically valid candidate f:

```text
l_f = sum over boundary bonds b of g(local_environment(b, f), acquisition(x))
c_f = r(contextualized_candidate_token_f)
c_centered_f = c_f - mean over valid candidates of c_f
s_f = l_f + c_centered_f
P(f | x) = softmax over valid candidates of s_f
R(x) = mean over valid candidates of c_centered_f ** 2
loss = existing spectral cross-entropy + lambda * mean over spectra of R(x)
```

Centering removes the component that already cancels in the softmax. The penalty acts on the
actual contribution to score differences, so rescaling a hidden layer or a learned mixing
coefficient cannot evade it. Use no learned mixing coefficient in this pilot. This is a soft
penalty, not a hard bound or a proof that context cannot dominate particular decisions.
It suppresses useful discrimination as well as potentially memorized differences, including
separation of strong fragments from implausible candidates. Lower context variance does not
establish reduced memorization; the early context-only and calibration controls are essential.

Evaluate the penalty after invalid candidates are removed, weighting spectra equally rather
than allowing larger candidate sets to dominate. Preserve the current within-fragment channel
probabilities and normalization over the surviving candidates. Uniform candidate variance can
be driven by low-probability tails. Taking a mean avoids linear scaling with candidate count,
but does not make the statistic invariant to changes in support. Keep this definition fixed
for the first pilot and report R, its top/tail contributions, context RMS, and actual penalty
loss by NCE and candidate-count bin. Do not add a weighting sweep after confirmation. A later
probability-weighted variant would use detached baseline weights; target-weighted fragment
variance would require ambiguous peak attribution and is not part of this pilot.

For a frozen linear readout c = w*t, R = w^T S w, with S the uniform within-spectrum token
covariance. A local quadratic approximation gives (H + 2*lambda*S)w = H*w_star when the
unpenalized solution is stationary and the local branch is absent/fixed. H is the actual
spectral-loss Hessian; mass aggregation means it need not be the simple candidate-softmax
covariance. If S is proportional to H, the change is pure score shrinkage (temperature);
otherwise it is direction-dependent shrinkage. This motivates the calibration controls and
is not a proof that every fitted penalized model is equivalent to temperature scaling.

The local branch starts with a shared small encoder of radius-two cut neighborhoods, width 32,
followed by an MLP. These are proposed pilot settings, not measured optima. Use raw element,
formal charge, H count, bond order, and explicitly documented aromatic/ring attributes; audit
their computation before implementation. Exclude learned precursor embeddings, partial charges
computed from the whole molecule, whole-molecule mass/formula, complement pools, candidate-set
attention, and proposal probabilities. Condition on NCE, instrument, and adduct using a separate
small acquisition encoder. Adduct is constant for this `[M+H]+` pilot and carries no variation.
`a_charge` is a whole-molecule Gasteiger charge and must be excluded. `b_conj`, `a_ring_size`,
and aromatic/ring flags use precursor-wide chemical perception; if retained, label them as
inherited descriptors rather than claiming they were computed strictly inside the neighborhood.
Preserve retained/lost endpoint orientation, so complementary fragments need not receive
identical scores. No sanitized product chemistry is implied.

For the zero-cut precursor, use a learned acquisition-only survival score instead of an
accidental empty sum of zero. Molecular effects on survival can still enter through c_f.
Use the same survival treatment in every arm containing a local branch. Read L both on the
whole spectrum and on independently renormalized nonprecursor contributions, restricting the
latter to spectra with measured fragment support and reporting excluded counts. Apply the same
precursor mass exclusion and normalization to every comparator. An L failure limited to survival
is not evidence that local fragment ranking lacks information. Fit L's fragment-only diagnostic
objective separately if survival dominates its training loss; predeclare this variant before
confirmation, and never present it as a full-spectrum model.

Summing boundary scores introduces additive cut-count effects and cannot describe arbitrary
interactions between remote cuts. Include a cut-count/acquisition-only replacement control
before attributing a gain to local chemistry. Radius-two neighborhoods and their aggregate can
still be distinctive; locality constrains inputs, not their statistical identifiability.

## Stage 1: optional allocation probe on fixed representations

Use the two local `i1` anchors, which were trained on the same 3,000-molecule manifest:
`ckpts/local_i1_ctrl/model-epoch=021.ckpt` (seed 397623) and
`ckpts/local_i1_ctrl_s2/model-epoch=023.ckpt` (seed 811237). Metadata, checksums and
reproduction are recorded in the [execution state](#implementation-and-execution-state-2026-09-29).
The full-data `i1` (`i50ig7rf`, epoch 59) is a third producer, read separately as the full-data
token check under the full-data split rule in Stage 2 item 3.
The [local-screen finding](../../findings/masster/masster_fragment_connectivity_local_screen_2026_09_29.md)
records train-eval cosine 0.8914/0.8954 and prior selection on local validation. These are local
anchors, not the full-data `i1` run. Pair every arm within each anchor and report the two producer
seeds separately; they supply backbone variation, unlike two refits of one checkpoint.

Cache candidates, valid channels, channel probabilities, and final contextualized candidate
tokens in evaluation mode. Freeze all producing modules, including encoder and attention
buffers. Freezing only proposal or channel-head parameters is insufficient: a changing shared
encoder would change their outputs. The context scorer is a trainable copy of the checkpoint's
scalar allocation head (`nn.Linear`) on these fixed tokens. Initialize it from that checkpoint
and initialize the final local projection to zero in the joint-fit arms, preserving their initial
predictions. U can receive weak local gradients on already well-fitted training tokens; it is
not by itself a reliable test of whether local chemistry contains complementary information.

Fit only this scalar context head and the new local branch, using observed spectra and the
existing peak-level loss. Never manufacture isomer-specific labels from ambiguous peaks.
This is a conditional allocation experiment: it can refit a linear readout but cannot change
how encoder or attention representations use local chemistry. A null does not rule out joint
representation learning, and a positive result does not establish an end-to-end gain. This
probe is optional and is not a prerequisite for stage 3A. Choose whether to run it before
starting fits; a null does not automatically launch additional experiments.

| Arm | Score / fit | Purpose |
| --- | --- | --- |
| B | Frozen checkpoint | Reproduction and no-fit anchor |
| C | Context head refit, no local branch, lambda = 0 | Matched refit control |
| U | Joint-fit local + context, lambda = 0, local projection starts at zero | Unrestricted joint-fit control; a null does not establish absence of local value |
| R1 / R2 | Local + context, weak / stronger penalty | Main context-restriction test |
| G1 / G2 | Context only, weak / stronger penalty | Generic regularization control, included from the start |
| T_gamma | B plus fitted `P^gamma` | Always-run temperature control |
| T_nce | B plus NCE-binned temperature and precursor offset (`ctrl_nce`) | Always-run calibration/survival control |
| L | Local scorer independently trained against measured peak-level targets | Local information and underfitting diagnostic, including fragment-only read |
| S | Frozen B and independently trained L, stacked on unseen tuning molecules | Complementary local signal without requiring residual gradients on memorized fit tokens |

Fit L independently on the fit molecules using the spectral objective and measured intensities,
not residuals or pseudo-labels from B. Stack its frozen scores with B using coefficients fitted
on unseen tuning molecules, including the same temperature/precursor-offset flexibility as
T_nce. S measures complementary signal; U measures joint optimization. Use
[scripts/eval/bond_local_learner.py](../../../scripts/eval/bond_local_learner.py) as a precedent,
not an unchanged launcher: its existing whole-molecule features, equal-split bond attribution,
mean-over-cuts aggregation, group metric, and val/test cross-fitting differ from this plan.
Any legacy replay is a separately labelled diagnostic, with no tuning on test or confirmation.

Construct two nonzero penalty strengths using training data only: scale their initial penalty
contributions to 1% and 10% of the initial spectral loss on a fixed training calibration sample.
Record resulting numeric lambdas and keep them fixed. This is only an initialization heuristic;
verify actual context RMS falls during fitting. If the baseline penalty is zero, investigate
the score extraction rather than silently substituting a denominator. Select the best strength
separately for R and G on the unseen-molecule tuning set defined below; do not force G to use
R's chosen strength. Include zero as a candidate through U and C, respectively. Give both
families the same search and stopping budget. Fit calibration parameters on tuning only.

Before attributing a gain to local chemistry, also run local+context with local chemistry
replaced by cut count and acquisition inputs. Select its strength independently using the same
zero/weak/strong search and exposure budget. Schedule this conditional control from tuning
results, before opening confirmation results. Lock all choices and fit per-arm calibration on
tuning only; the confirmation set must not trigger another search.

## Stage 2: validation and decisions

1. **Reproduce before fitting.** Cached reconstruction must match checkpoint fragment/channel
   probabilities and production spectrum metrics within recorded numerical tolerance. All arms
   must have identical candidate membership, masses, valid-channel masks, and coverage.
2. **Tune on molecules unseen by all producing models.** For the local anchor pair, fit only
   on the 3,000 identities in `output/local_encoder_noise_2026_09_27/split/train_ids.csv`.
   Tune on canonical `nist20_inchikey` training identities outside that set: the CSV census
   gives 12,668 minus 3,000 = **9,668 identities** before runtime filters. Confirm on the
   1,000 local validation identities; these have no overlap with canonical training identities.
   The fit/confirmation CSVs contain 37,111/12,381 spectrum rows. Record eligible counts after
   filtering and keep every CE row of an identity together. Local `test_ids.csv` is a training
   subset, not an independent test. Audit actual encoder/teacher training identities before
   accepting the split. Novelty is measured against the union of identities used to fit the
   producer and local scorer (the same 3,000 here), not the full canonical training set.
   The confirmation set previously selected checkpoints and appeared in prior analyses: label
   it reused development evidence, not an untouched benchmark.
3. **Cross-fit tuning and match exposure.** Use fixed molecule folds within the 9,668 tuning
   identities for out-of-fold calibration/stacking comparisons. Select penalties and stopping
   there, then refit only calibration/stacking coefficients on all tuning identities before a
   single confirmation read. Include T_gamma and T_nce for every producer from the outset and
   calibrate each neural arm under the same protocol. Never cross-fit using confirmation or
   test labels. Report both producer seeds with identical per-anchor fit seeds and sampling;
   repeat a fit seed only if optimization variability needs resolution. Use equal search and
   stopping budgets, set maximum epochs from control convergence on tuning, and report
   nonconvergence rather than selectively extending a favored arm. A full-data `i1`/`i4`
   anchor has already seen the 9,668 identities: for such an anchor, use disjoint canonical
   validation tuning/confirmation manifests instead and disclose historical reuse. Do not
   transfer the local split's unseen status or novelty reference to a full-data model.
4. **Read performance.** Report production cosine, calibration-controlled cosine, spectral CE,
   base-peak hits, and nested/sibling balanced accuracy using the existing pair definitions.
   Slice by nearest-training-molecule similarity (<0.4, 0.4-0.6, >=0.6), NCE, instrument,
   cut count, and precursor/nonprecursor outcomes. Report support counts and molecule-clustered
   paired uncertainty. Keep the primary spectral aggregation identical to the baseline.
5. **Read capacity separately.** Measure train and validation in eval mode, loss curves, local
   and context score RMS, their correlation, and their A-vs-B differences. If both train and
   validation degrade and the unrestricted control has converged, stronger restriction likely
   removes useful capacity. Check optimization before labeling this representational underfit.
6. **Evidence gate and minimum effect.** Primary delta is calibrated **production** cosine
   against the stronger of T_gamma and T_nce, with control choice locked on tuning. Adopt the
   older 2A numerical bars as a new decision rule here, not as metric equivalence to its group
   cosine: pass requires at least **+0.005 in each producer seed**; below **+0.002 in either
   seed** is insufficient for promotion; all remaining cases are inconclusive. Also require
   superiority over B, independently calibrated C/U/G, a positive molecule-bootstrap 95%
   lower bound, and consistently improving unfamiliar nested/sibling ranking. The cut-count
   control must not explain a chemistry-specific claim. Compare R with S: if S matches it,
   prefer the simpler independently fitted local correction and do not claim the penalty is
   necessary. Report raw production cosine and require no negative paired mean versus B;
   neither raw-only gains nor significance at +0.001 pass. Report subgroup regressions and
   producer-seed disagreement. Underpowered results remain inconclusive, not evidence of no
   effect. This gate supports this recipe only; the optional frozen-token result does not
   determine whether a separately bounded representation-learning pilot may be tested.

| Observation | Decision |
| --- | --- |
| Moderate restriction improves held-out allocation and beats B and controls | Evidence for the allocation recipe; joint-learning and production claims still need stage 3 |
| S improves but R1/R2 do not | Independently trained local chemistry complements the model; no evidence for restricting context |
| U is null but S improves | Joint residual fitting failed to expose available local signal |
| L loses, moderate restriction helps | Keep context; reject local-only replacement |
| Context-only penalty matches the gain | Prefer the simpler regularizer; no local-chemistry claim |
| Cut-count or calibration control matches the gain | Reject the chemistry-specific explanation |
| Stronger penalties hurt both train and validation | Back off; do not keep tightening to force locality |
| No reproducible gain in the optional frozen-token probe | Close that probe; leave the joint-learning hypothesis unresolved, with no automatic campaign |
| No reproducible gain in the bounded joint-learning pilot | Stop this scorer recipe; no automatic deeper encoder or flow campaign |

## Stage 3: joint representation learning and production confirmation

### Anchor choice (user decision, 2026-09-29)

The full-data token check and 3A use `i1` (`i50ig7rf`, epoch 59): its checkpoint is local and
every diagnostic this plan reads was measured on it. `i3` and `i4` exist only on W&B. The stronger
lineage is therefore required only before 3B: repeat the allocation comparison on verified `i3`
(or `i4`) tokens, including T_gamma, T_nce, G and S, before any 3B job. The route plan records
`i3` +0.0102 over `i1` from key dropout plus EMA, and `i4` +0.0019 over `i3` on one seed (inside the
0.0042 seed spread). `k1` key dropout regularizes the same boundary-encoder/token route this
penalty targets, so a gain on `i1` may be one `k1` already provides. A win over `i1` alone is never
promoted. A frozen-token null does not forbid a separately justified local joint-learning pilot,
but supplies no evidence for cluster escalation.

### 3A. Bounded representation-learning pilot

This is the decisive test of whether training representations with a constrained context
contribution helps transfer. It may be the first fitting experiment, or follow the optional
stage-1 probe regardless of that probe's outcome. Record the chosen route and fixed pilot
budget before launch; neither this plan nor a probe null authorizes a training campaign.

Use the `i1` configuration and checkpoint (`i50ig7rf`, epoch 59, md5
`d70f21d8109ebde2fbd817dc261898fe`, raw weights, no EMA). Before launch, record whether 3A
fine-tunes those weights or trains from scratch with that checkpoint only as teacher; a
fine-tune tests whether the penalty can undo routes an already-trained encoder has learned, not
whether representations learned under the constraint transfer. Train the allocation encoder and
attention as well as the scalar head and local branch. For isolation, use an immutable copy of
that checkpoint as the proposal and ion/H teacher for all arms of that producer seed.
Cache or run that separate teacher; freezing heads sharing the trainable encoder is insufficient.
Carry over the C, U, R, G, S and calibration comparisons and underfitting diagnostics, with independent equal-budget
penalty selection. B is the untouched initialized model reconstructed on the same teacher
support, with any discrepancy from ordinary `i1` measured before fitting. Use the unseen tuning
and confirmation split rules above for every pretrained model and teacher.

Call C here the **teacher-constrained `i1` control**. It is a modified training system, not
ordinary production `i1`. Passing the evidence gate against this control and B supports joint
allocation learning only. Record coverage/channel equality, compute and memory, and whether
any benefit survives calibration. If the bounded pilot fails, stop this recipe rather than
expanding the encoder, penalty search, or graph design automatically.

### 3B. Conditional integrated-model confirmation

Only after 3A succeeds and the `i3`/`i4` token check above has run, test the integrated model
with normal shared-system training, stacked on `k1` plus EMA, and compare it against a fresh
**ordinary `i3` control** (or `i4`, for the best recorded tail, 0.7729), the unrestricted
local+context model, and the independently tuned context-only penalty control. Match data,
initialization recipe, dropout, EMA, normalization, seeds, exposure, and tuning budget. The
ordinary control must retain its normal proposal and ion/H behavior; do not replace it with the
teacher-constrained control or compare only with historical W&B numbers.

Apply the evidence criteria on locked confirmation data, with the ordinary `i3`/`i4` control as a
mandatory baseline. Track changing candidate coverage and ion/H allocation; these are now part of
the integrated system and cannot be assumed constant. Re-audit other identity routes and report
backbone seed variability. A win over frozen `i1` or teacher-constrained `i1` alone is
insufficient for production promotion. No whole-model invariance is implied by a scalar
allocation penalty.

## Storage and compute budget

Store token caches under `output/local_context_allocation/`, keyed by checkpoint, spectrum and
config identity. Tokens depend on CE and candidate-set attention, so do not cache one token set
per molecule. At 256 candidates x 256 dimensions, fp32 tokens alone use 256 KiB per spectrum:
37,111 fit spectra require about 9.73 GB decimal and 12,381 confirmation spectra about 3.25 GB
per producer. Tuning spectra and channel/topology arrays are additional; two producers multiply
storage unless processed sequentially. These are upper-bound token arithmetic, not measured
cache sizes. Audit actual candidate counts and free disk before writing; do not put caches in
`/tmp` (3.7 GiB filesystem on this workstation). fp16 halves token storage but is allowed only
after measuring baseline reconstruction error and recording tolerance. Stream/chunk tuning
records or process one producer at a time if the full cache exceeds available space.

No runtime estimate is established for this architecture. Before any fit campaign, measure
cache generation, one epoch and forward/backward peak memory on representative batches for
baseline and new arms, including teacher cost. Record projected total GPU-hours for the full
arm/seed/search budget and set a fixed wall-clock ceiling. Report matched-epoch results and
best-by-hour curves, choosing checkpoints by tuning only; confirmation is read only for locked
checkpoint choices. Include caching, local pretraining and tuning overhead, not just final-head
fit time. These measurements are required before cluster scheduling.

## Implementation checks and handoff

When implementation is requested, read the Python rules and introduce default-off config keys
through the existing schema. Preserve checkpoint compatibility, output shapes, and baseline RNG
initialization. Focused tests must cover disabled-path reproduction, candidate permutation and
batch isolation, finite gradients, zero-cut survival, invalid-channel removal, normalization,
and loss invariance to per-spectrum context offsets. Verify local scores remain equal when
their extracted local inputs and acquisition are equal despite different remote structures;
this applies to local scores, not final probabilities. Verify frozen teachers and candidate
sets stay unchanged after optimizer steps, and penalty gradients reach only the intended path.

Record exact feature whitelist, feature provenance, checkpoint/cache identities, split manifests,
numeric lambdas, seeds, runtime/memory, and commands before any fit. Use the configured
`FRAGNNET-GPU` interpreter. No new datasets, packages or scheduler jobs are part of Stage 1.

## Implementation and execution state (2026-09-29)

Stage 1 is implemented as a debug probe; the model code is unchanged (no config keys). The user
approved the untouched read and the cache clean-up on 2026-09-29.

- **Code.** [`debug_masster_local_context_allocation.py`](../../../scripts/debug/masster_memorization/debug_masster_local_context_allocation.py)
  with subcommands `manifests`, `extract`, `fit` and `report`;
  [tests](../../../tests/test_debug_masster_local_context_allocation.py) (14) cover the segment
  ops, penalty offset invariance, compact-bin cosine against `cos_sim_helper`, manifests,
  config merging, zero start, retained/lost orientation, radius-2 locality, batch isolation,
  penalty gradients reaching only the context head, CE/cosine offset invariance, identical local
  initialization across arms, and the capture/rebuild of a real tiny MASSter forward.
- **Producers.**

  | producer | checkpoint | md5 | fit / tune / confirm molecules (spectra) |
  | --- | --- | --- | --- |
  | A1 | `ckpts/local_i1_ctrl/model-epoch=021.ckpt` (seed 397623) | `cb871c351e8a06fd5d2ac44e5c6ecdb1` | 3,000 (37,111) / 9,668 (120,567) / 1,000 (12,381) |
  | A2 | `ckpts/local_i1_ctrl_s2/model-epoch=023.ckpt` (seed 811237) | `5deb1aa801b5dfc3574161d2ce811d64` | same |
  | F | `saved_ckpts/i1_inst_ob_obf_i50ig7rf/ckpt/model-epoch=059.ckpt` | `d70f21d8109ebde2fbd817dc261898fe` | 3,000 (37,111) / 3,222 (39,639) / 1,000 (12,381) |

  F's tune set is canonical val minus local val. Every producer also has an untouched read on
  the canonical test split (4,223 molecules, 52,621 spectra), which none of them saw in training
  or checkpoint selection. **Disclosure:** local val (confirm) selected the A1/A2 checkpoints,
  and canonical val (F tune plus confirm) selected F's; those reads are reused development data.
  Novelty is taken against the local 3,000 for A1/A2 and all 12,668 canonical train molecules
  for F. No spectrum is lost to dataset filters.
- **Config.** Checkpoint hparams win; keys a checkpoint predates are filled from the template
  plus `i1_inst_ob_obf.yml` and recorded per cache (for example
  `masster_fragment_connectivity_layers`, and `masster_intensity_mass_window` for F).
- **Reproduction.** Each cache rebuilds the model's output from captured pre-filter tensors:
  channel log-probabilities within 7.6e-6, CE within 5.7e-6, `cos_sim_0.01` within 1.6e-6 on
  every spectrum. The mean `cos_sim_0.01` reproduces each checkpoint's logged selection value:
  A1 confirm 0.66750 (logged 0.66750), A2 confirm 0.67015 (0.67016), F over the full canonical
  val 0.76219 (0.76218). Local tuning molecules score like validation (A1 0.6672, A2 0.6725).
- **fp16 tokens.** The checkpoint head on fp16 tokens differs by at most 2.4e-3 in a logit and
  2.4e-4 in a spectrum's cosine; arms start from the fp16 tokens, B uses the exact logits.
- **Deviations from the text above.**
  - *Penalty strengths.* The production CE is a Gaussian-density cross-entropy with an arbitrary
    offset and is negative (A1 L0 = -3.43), so "1% / 10% of the initial loss" is read offset-free:
    of `L_uniform - L0`, the loss the checkpoint head removes relative to a uniform allocation.
    A1: `L_uniform` -0.578, R0 5.83, lambdas 0.0049 and 0.049.
  - *Calibration.* T_gamma, T_nce and S are fitted by maximizing the tuning production cosine
    (differentiable, on the cached bins) on a fixed molecule subsample (1,500 per fold, 3,000
    for the final fit); each arm's reported family is its better one by two-fold (molecule
    parity) out-of-fold cosine, and strengths within a family are compared under T_nce.
  - *Stopping.* Every fitted arm trains in lockstep on identical batches (Adam 1e-3, batch 256,
    at most 30 epochs) and stops on tuning CE with patience 4; the arms of one structure share
    their initial local weights.
  - *Pair reads.* Relation labels are fixed from each checkpoint's own dominant structures; the
    primary nested/sibling read uses unambiguous pairs, with the rate at which an arm's dominant
    structures differ from B's reported.
  - *Local features.* Atom: element, formal charge, H count, in-ring/aromatic (inherited);
    bond: type including aromatic, in-ring (inherited). Excluded: degree, hybridization,
    radical, ring size, mass, chirality, Gasteiger charge, conjugation.
- **Cost.** Extraction runs at about 190 spectra/s; caches take about 181 KB per spectrum (fp16
  tokens 117 KB), 86 GB in total.
  A1 fitting takes 7.6 min per lockstep epoch (seven arms, 120,567 tuning spectra read each
  epoch); ceiling 4.5 h per local producer.
- **Stopped by the user (2026-09-29, 20:57).** The A1 fit reached epoch 17: every context arm
  had early-stopped (C and U best at epoch 2, G1/G2/R1/R2 at epoch 1) and only L was still
  improving. No arm state, lock or report was written, so no calibrated read and no confirm or
  untouched read exists. A2 and F were never fitted. A1's tuning history is in
  `fits/a1/history_context.csv`. Raw tuning cosine at the best epochs (not the gate): B 0.6672,
  C 0.6705, G1 0.6707, G2 0.6673, U 0.6708, R1 0.6708, R2 0.6677, L 0.49 and rising. Context
  RMS fell from 2.41 to 1.99 (G1) and 1.39 (G2); the local term stayed small with a free
  context (U, RMS 0.09, correlation -0.20 with it) and grew under the penalty (R2, 0.57,
  +0.29). At the user's request the cache arrays were deleted the same day (77 GB); each cache
  keeps `provenance.json` and `spectra.pkl` (per-spectrum model cosine and CE). Splits,
  novelty and scripts are kept under `output/local_context_allocation/` (Git-ignored).
  Resuming needs `run_extract.sh` again (about 45 min), then `fit` per producer, the untouched
  extraction and `report` (`run_fits_after_a1.sh`, `run_untouched_and_reports.sh`).

Next: the user is redirecting to ORB/ORB_F reruns of the route-removal arms (complement pool,
masked set attention); resume or close this probe after that decision.
