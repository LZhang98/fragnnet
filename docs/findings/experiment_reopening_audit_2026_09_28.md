# Which experiment directions are worth reopening?

> **Status:** FINDING · **Updated:** 2026-09-28
> Research-record and source audit; no new training, inference, or live W&B refresh.
> Recommendations below are hypotheses, not promoted experiments. Historical closures stay intact.

## Decision

For a substantial single-model improvement on the existing MASSter data, the most defensible
architecture experiment is **message passing inside the retained candidate fragment**, added to
the current model rather than replacing its molecular context. This reopens the representation
question; it does not revive FluxMPNN or claim that a post-cut GNN already passed a gate.

The next research options are **cross-molecule supervision** and **ensemble distillation with
out-of-fold teachers**. Both survived the audit because the relevant experiment was not actually
settled by the cited nulls. Their benefit remains unmeasured. No record establishes an inexpensive
change that will recover a large fraction of the remaining gap.

Two narrower reconsiderations have direct new evidence: smooth CE input after instrument
correction, and R-Drop after the eight-view averaging follow-up. These are bounded questions,
not leading explanations for the large ranking deficit.

## Scope and evidence standard

- Screened the inventory, status, and recorded outcomes of all **125 Markdown records** under
  `docs/plans/`, `docs/findings/`, and `docs/notes/` present before this audit. Read the detailed
  results and follow-ups for the candidate reopenings and their counter-evidence. The curated
  missing-peak logs were screened by cause and summary; they were not re-annotated molecule by
  molecule. Engineering/performance records were screened for experiment confounds and cost.
- Read current project state, architecture, the memorization/campaign/architecture ledgers,
  and the new `i1` base-peak, residual, and instrument findings. Later dated results take
  precedence over stale status text in an older section.
- Inspected historical branch trees without switching branches, recovering the missing roadmap,
  directional-cut encoder, static-fragment DAG model, FraGhost, pairwise-loss/sampler design,
  recent-change log, and formula-bag correction diagnostic.
- Checked the current MASSter candidate pooling/head path and auxiliary-loss configuration.
  This is not a complete code audit or a replay of every historical run.
- No live W&B histories or cluster stdout logs were downloaded. Numerical results are the
  recorded measurements; running/unread jobs require a fresh read before scheduling follow-ups.
  The local tree already contained unrelated uncommitted documentation, probes, and tests.

Audit source HEAD: `3d5a99dfcb2bf016e68d55642697f7c2f0f80be5`, plus the pre-existing working
tree. Historical sources, recoverable with `git show <revision>:<path>`:

| Revision | Historical record |
| --- | --- |
| `68046f29b1aa8ccc494265955577c255bff87d58` | `docs/directional_cut_encoder.md`; `docs/model_improvement_plan.md` |
| `4c46059752febde138359de0a645fa56d7386049` | `docs/static_fragment_dag_gnn_plan.md` |
| `57727e02a8c5a17468fa785075c0132a98aaf1aa` | `docs/formula_bag_ce_response_curve_diagnostic.md`; `docs/pairwise_loss.md`; `docs/recent_changes.md` |
| `68882581450b9a512b4e6cc580f55d5364d10a62` | `docs/FraGhost.md` |
| `36bd150758c65d9b2e3261f6b6696a9e3f81d012` | `docs/plans/model_improvement_plan.md` |

## What the newest evidence changes

The [base-peak finding](masster/masster_i1_base_peak_errors_2026_09_28.md) is more selective
than the earlier verbal diagnosis:

- Nested candidate pairs account for 60% of base-peak misses; overlapping/disjoint alternatives
  account for another 34%. Complementary charge choices account for only 1.4%.
- The ranking deficit is 0.104 on validation versus 0.010 on the sampled training molecules.
- Eight chemistry-attribute corrections add only +0.0004 beyond temperature, or +0.0009 when
  fitted per NCE bin. Aggregate calibration is already close; the missing behavior is
  discriminating between candidates on new molecules.
- Nested-pair discrimination follows novelty and is approximately flat across NCE. Containment
  is a structural relation, not an observed reaction pathway. Its frequency does not establish
  sequential kinetics as the cause of the error.

The [instrument result](nist_orbitrap_instrument_ce_scales_2026_09_28.md) invalidates a blanket
interpretation of older identity-route probes: part of their train-specific signal carried an
omitted instrument label. It does not invalidate every earlier null. Reopen a direction only
when that omission is relevant to its mechanism or there is another concrete change in the test.

## Ranked research shortlist

### 1. Reopen fragment representation through retained-subgraph message passing

**Previous record.** Static fragment descriptors, directional cut encoders, bond-set pooling,
and candidate adjacency were explored or proposed. The
[cut-set screen](../plans/masster/masster_cut_set_features_plan.md) found no promoted feature
arm: adjacency +0.0032, raw cut-bond features -0.0010, their combination +0.0021.
The local boundary encoder `r1` gave -0.0029. These do not test a GNN running on each retained
fragment's internal bonds. Historical static-fragment code describes static aggregates plus a
GNN **between DAG nodes**, which is also a different operation. The recovered directional-cut
record has an implementation description, not an attributable training outcome; do not label it
a measured failure or success.

**What would change.** Keep the precursor encoder, candidate generator, oriented cut features,
and existing allocator. Add a small residual branch that processes the retained atom-induced
graph with cut-site markers and produces a candidate representation. It describes connectivity
after removing the complement. It does not predict reaction flux, build a new fragmentation DAG,
or infer a unique protonated/rearranged product structure.

**Why reconsider.** The latest failure table identifies relevant decision populations, and the
current [model](../../src/fragnnet/model/masster_model.py) pools precursor-contextualized atoms
into fragment/complement tokens before the intensity head; it does not perform candidate-specific
message passing on retained internal bonds. This is a sample-efficiency/inductive-bias hypothesis,
not proof that information is absent from the existing token.

**Smallest decisive test.** First price assembly, memory, and forward/backward time on actual
candidate batches. Reuse unique atom sets across CE where possible without changing proposal
selection. Compare `i1`, a pooled branch with matched extra capacity, and the retained-edge
branch with the same node inputs and pooling. Preserve the baseline initialization and candidate
support. Initially the branch must be small; no promised runtime or accuracy improvement.

**Gate.** Replicated validation cosine gain, including calibration-controlled readout, plus
better nested/alternative-candidate decisions below Tanimoto 0.6. Report all-molecule quality,
coverage, and cost. A smaller train/val gap caused only by worse train fit does not pass. If only
training improves, or the matched pooled branch performs equally, the connectivity claim fails.

### 2. Resume cross-molecule supervision, not another within-spectrum ranking loss

**Previous record.** The historical `pairwise_loss.md` says ordinary batches are dominated by
unrelated spectra with near-zero off-diagonal similarity. It proposed informative grouping but
does not record a successful matched run. The
[generalization note, section 7](../notes/masster_generalization_methods_2026_09_27.md)
independently proposes target-distance supervision. This is unfinished work, not a reversed null.

**What would change.** Train a small auxiliary representation to preserve measured spectral
relationships across **different training molecules** under matched acquisition conditions.
Retain the normal spectral loss. Do not force all spectra with the same formula or local motif
to be similar; actual measured differences must remain learnable.

**Why reconsider.** Most prior arms penalized identity information without specifying what should
transfer between molecules. Near-perfect within-spectrum train ranking does not answer this
cross-molecule question. It also does not prove that another ranking loss must fail; it simply
weakens the rationale for one aimed at fitting the already-solved training ordering.

**Smallest decisive test.** A train-only census of informative, nontrivial spectrum/neutral-loss
neighbors, controlling instrument, adduct, CE, precursor dominance, and mass. Require signal
beyond random matched pairs and trivial formula identity before building a trainer. If it passes,
compare the auxiliary objective against the same sampler without that objective, and shuffled
relationships. Do not let a different molecule/CE sampling budget masquerade as a loss gain.

**Gate.** Better prediction on held-out molecules, not merely a better embedding-neighbor score.
The earlier `sp0` sampler null does not test this: it groups energies of the **same** molecule.
The corrected CE-context sampler remains relevant implementation evidence, not efficacy evidence.

### 3. Reopen ensemble distillation, preferably with out-of-fold teachers

**Previous record.** The clean-target cross-evaluation (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_target_cleaning_cross_eval_2026_09_23.md`)
found +0.020 from averaging f0/f1 predictions. The novelty analysis (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_variance_novelty_2026_09_24.md`)
found larger gains on unfamiliar molecules. Those models differ in target cleaning as well as
their learned solutions, so first establish the gain with a pure seed ensemble on current labels.

**What would change.** For student training molecules, obtain predictions from teachers that
excluded those molecules during fitting. Combine teacher supervision with real measured spectra,
aligned at observable peak masses; do not align latent fragment row IDs. All teacher folds and
any unlabeled teaching molecules must respect the downstream validation/test exclusion contract.

**Why reconsider.** Switch EMA `u3` (-0.0089 versus ordinary EMA) feeds averaged weights back into
the same training trajectory. It is not an out-of-fold teacher/student experiment. Likewise,
the fact that an in-sample teacher memorizes labels does not close out-of-fold distillation.

**Smallest decisive test.** Establish teacher complementarity on a development split and inspect
teacher error on molecules excluded from fitting. Use a bounded fold pilot before paying for
many teachers. Compare a student of the same architecture/training budget without the teacher
term. Account for teacher mass outside student support instead of renormalizing it away silently.

**Gate.** Retain a meaningful part of a measured ensemble gain in a single predictor. Teacher
cost is substantial, and a weaker out-of-fold teacher may hurt; there is no established gain on
`i1`. This is the option with a measured averaging benefit behind it, not a measured student benefit.

## Narrow or conditional reopenings

| Direction | Why its old verdict is incomplete | Next check and limit |
| --- | --- | --- |
| Smooth CE input (`n1` -> `i2`) | The categorical NCE code also identified the instrument. After `OB`/`OB_F`, the early penalty shrank from -0.021 to -0.004. | Read the completed `i2` trajectory against `i1` before anything new; test held-out CE interpolation separately. The recorded comparison is mid-run, not a final promotion. This does not reopen a full QET flow head. |
| R-Drop | The original two-view argument overstated how much averaging was just flattening. Eight views add +0.0031 beyond flattening on `i1`. | A matched two-pass supervised control is essential. The signal is small and not novelty-specific; keep below the main shortlist for a large improvement. |
| Full attention with four heads / m/z encoding | Local `fa_h4` was +0.0086; the added-m/z run was stopped before its planned tail. Merged eight-head full attention was only about +0.003. | Confirm the exact configuration on current `i1` data if no later result exists. Different recipe, seed, and stopped-run selection prevent promotion. The local atom-mask gain already failed to reproduce on Fir. |
| Spectrum EBM reranking | Proposed, never trained; the historical formula-bag correction null is not the same optimization/inference mechanism. | Run the existing structured-pool versus matched-random oracle gate before a scorer. Use out-of-fold predictions for a transfer-oriented training design. Training-fit residuals are not representative of novel-molecule errors. Preserve the user's recorded bond-cut-first diagnostic order unless changed. |
| ClassyFire/motif pretraining | Proposed, no transfer experiment recorded. Deterministic feature recoverability does not prove supervised pretraining useless. | Audit availability of atom-mapped labels first; molecule-level ClassyFire labels do not imply atom labels exist. Test a small compatible encoder with a scratch control. This is extra-data work and not independent QC knowledge. |
| ChemBERTa with an exact-mass candidate head | E1/T2 failed as a joint encoder plus dense 0.01-Da output system. No random-init or matched-head control isolated the pretrained encoder. | Reuse candidate masses with frozen/pretrained/random controls only if pursuing this track. Do not rerun the same dense head or assume a better head makes the encoder useful. |
| FraGNNet prepass with `num_cut_bonds` | `depth` was an incorrect cut-count proxy on multi-cut prepass DAGs; the correct feature was off in the recorded failed arm. | A matched prepass × cut-count comparison has a concrete confound to remove. This is FraGNNet-specific, not a missing MASSter feature. |
| Formula-bag CE correction | Historical coefficients became nearly uniform across bags and saturated, so they largely canceled in softmax. This did not exercise the intended differential correction. | A centered, nonsaturated parameterization merits at most a small gradient/learnability diagnostic. Removing a degeneracy supplies no new transfer signal; not a leading training recommendation. |
| Cross-adduct supervision | Data-side gate passed, model-side gate not recorded as complete. | Recheck with corrected instrument labels and a multi-adduct checkpoint; current `i1` is positive-mode only. Do not impose equality between different adduct spectra. |

Canonical sources: [CE/QET](masster/masster_qet_projection_candidate_union_2026_09_26.md),
[instrument correction](nist_orbitrap_instrument_ce_scales_2026_09_28.md),
[dropout follow-up](masster/masster_dropout_view_disagreement_2026_09_27.md),
[local attention screen](../plans/masster/masster_memorization_route_plan.md),
[EBM gate](../plans/energy_based_spectrum_reranking_plan.md),
[ClassyFire plan](../plans/fragnnet/classyfire_atom_pretraining_plan.md),
ChemBERTa results (removed 2026-10-14; `git show 8a3cc7a5:docs/plans/chemberta3_msgym_plan.md`),
[prepass confound](feature_headroom_2026_09_22.md),
[cross-adduct gate](cross_adduct_loss_headroom_2026_09_17.md).

## Keep closed or deprioritized for the current objective

| Family | Evidence that still applies |
| --- | --- |
| Unsupervised FluxMPNN / conserved formation-survival flow | Trained null-to-harmful; local branching lookup and latent identifiability remain. The sibling diagnostic measured node marginals, not observed edge flux. Nested candidate errors do not reverse this. |
| Complementary charge supervision and another charge-feature bundle | Complementary pairs are only 1.4% of current base-peak misses; basicity correction is null. Gasteiger `c_charge` is already in the token. Earlier recommendation to add it was redundant. |
| Generic local lability / tied cut descriptors | Offline learner about +0.002, almost no gain above NCE 60; `r1` and raw cut-feature arms do not establish a larger effect. Peak-level retraining of the local learner remains a bounded unfinished gate, not a major solution. |
| Simple cut adjacency / ring topology / composition | These arms ran and were null or below promotion gates. They were not merely forgotten configs. |
| More identity removal, stronger general dropout, smaller allocator | Removing boundary/global inputs did not reliably improve validation; allocator dropout `h1` hurt. `sm1` is a completed null, not an unrun idea. |
| More weight decay / longer training / cosine schedule stacked on EMA | Unmerged `w1` null; long-horizon gains tiny; `s1_k1_ema` -0.0011 versus `k1_ema`. Merged `wd10` remains incompletely tested but is a low-priority bookkeeping gap. |
| Switch EMA | Completed underfit failure versus ordinary EMA; retain ordinary EMA. |
| Global sharpening, cut-count/depth priors, extra acquisition offsets | Sharpness flips with novelty; broad attribute corrections add almost nothing. Recorded acquisition fields and precursor rescaling do not recover meaningful cosine on `i1`. |
| Full QET curve restriction / CE-to-depth law | Predictions already approximately satisfy the curve; projection only about +0.002. Output smoothness and missing chemical discrimination are different questions. |
| CE-blind shared proposals as currently built | Resumed `sp1` still loses 0.018 coverage and -0.0083 cosine versus `sp0`; the sampler fix did not rescue the proposal design. A support-preserving redesign would be a new efficiency experiment. |
| Escape head, target cleanup, generic false-peak penalties | Escape is a two-seed near-exact null; cleanup's apparent gain was easier evaluation targets; historical false-peak penalty harmful. These do not address the dominant ranking-transfer error. |
| More generic coverage, marginal selection, connected-fragment beam, cut EBM | Coverage is a small share of current error. Marginal selection failed; beam has a structural large-fragment hole. Cut EBM needs its own coverage/efficiency gate, not a claim that all missing intensity is proposal error. |
| Unrestricted SMARTS/excision/ChemFrag expansion | Generic excision fails its matched null; the supplied 32 ChemFrag transforms did not establish unique useful coverage. Restricted nitro NO loss passed a class gate but is too small to explain a large overall gain. |
| Isotope envelopes | Enveda's measured linear-cosine gate failed. NIST20/MSGYM or a different metric can reopen the dataset-specific gate, not erase the Enveda null. |
| Structural teacher based solely on NIST peak formulas | Formula labels do not distinguish same-formula substructures. The original MASSter teacher argument was explicitly withdrawn. |
| FraGhost/global companion as a default new model | Recovered proposal has no matched current-protocol gain. MASSter already has global and candidate attention; benchmark complementarity before building another large model. |

Sources: [FluxMPNN](../plans/fragnnet/fluxmpnn_plan.md),
[sibling diagnostic correction](../plans/fragnnet/sibling_ranking_target_plan.md),
[attempts audit](masster/masster_memorization_attempts_audit_2026_09_27.md),
[latest batch read](../plans/masster/masster_memorization_route_plan.md),
[architecture results](masster/masster_architecture_arms_epoch128_2026_09_22.md),
[shared proposals](masster/masster_shared_proposal_pathway_audit_2026_09_26.md),
[excision](masster/masster_excision_continuation_gate_2026_09_28.md),
[ChemFrag triage](masster/chemfrag_rearrangement_triage_2026_09_14.md),
[isotope gate](../plans/masster/masster_isotope_envelope_plan.md),
[teacher retraction](masster/masster_identity_supervision_2026_09_22.md).

## Corrections and execution boundary

- The earlier conversational formation/survival proposal repeats FluxMPNN's mechanism. It is
  not a new direction. The cheap charge-descriptor recommendation also overlooked existing
  inputs and the newer charge-choice null.
- The observed 18% bond-environment ICC is a population- and representation-specific
  measurement, not a mathematical ceiling on every possible QC descriptor. Conversely, fitting
  training spectra almost perfectly does not prove that the representation generalizes.
- The attempts ledger's old `s1`/`s1_k1_ema` “never submitted” row is superseded by the September
  28 batch read. Similar stale banners occur in feature and sampler records; this audit uses
  their later results. It does not rewrite historical result tables.
- The 0.104 ranking deficit and candidate-space cosine in the new base-peak finding are
  tolerance-matched; they must not be substituted for production `cos_sim_0.01`. The CPU/CUDA
  bin-grid discrepancy must be controlled in any new rescoring.
- Existing `i3`/`i4` are prepared experiments, not reopenings. NIST23 expansion is a separate
  data option; its +0.018 estimate is correlational. No numerical gain forecast is assigned to
  the research shortlist and gains from different lineages are not summed.

**Next concrete check:** price the retained-fragment branch against a matched pooled branch on
real `i1` candidate batches, and establish the train-only cross-molecule neighbor census before
choosing a training experiment. These are recommendations; neither was run in this audit.
No architecture, configuration, dataset, job, or historical closure was changed.
