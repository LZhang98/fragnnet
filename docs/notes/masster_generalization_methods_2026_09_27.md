# Further literature: practical generalization methods for MASSter

> **Status:** NOTE · **Updated:** 2026-09-28
> Primary-source search and repository cross-check following the
> [memorization review](../findings/masster/masster_memorization_plan_review_2026_09_27.md).
> No training launched. The MASSter adaptations below are proposals, not published results
> on this architecture. Existing run status comes from local records, without a W&B refresh.
> *2026-09-28:* the two-view disagreement gate ran on four weight sets
> ([finding](../findings/masster/masster_dropout_view_disagreement_2026_09_27.md)). It passes on
> magnitude but gives R-Drop no train-specific target. The centered-logit penalty should not be
> built as specified (§2). Engineering constraints were added to §3 and a prior to §6.
> A later §3 build review drops the cache and manual-optimization constraints for allocator-only
> SAM, adds radius scales and notes the missing branch path. No code changed.
> An eight-view follow-up corrects §1's "gain equals flattening": the flattened MC-dropout mean
> beats flattened eval on val by +0.0031 on `i1`, the new control.

## What changes after this search

The earlier [literature note](memorization_literature_2026_09_24.md) contains many method names,
but its ranking is dominated by removing molecule identity. A stronger next set tests prediction
stability and feature learning while retaining the chemistry. In particular, random dropout,
consistency between dropout views, and adversarial perturbations are different interventions.
Failure of one does not close the other two.

The first experiments worth pricing are peak-level R-Drop, a modest centered-logit penalty,
and a short SAM/ASAM continuation. Spectral decoupling, representation self-challenging, and
target-distance representation learning add ideas absent from the earlier memorization note.
R-Drop, FLAG and SAM were already mentioned but not developed into the experiments specified here.

All proposals use the current labeled molecules. Extra-data pretraining is outside the existing
scope. New papers do not justify launching all of these at once.

## 1. R-Drop on predicted peak distributions

**Paper.** Liang et al., *R-Drop: Regularized Dropout for Neural Networks*, NeurIPS 2021
([paper](https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html)).
It adds symmetric KL between two independently dropped-out predictions of the same input,
alongside both supervised losses. The reported tasks include NLP and image classification;
this is not direct evidence for MS/MS prediction.

**Why here.** Key dropout and averaging already help MASSter. R-Drop explicitly penalizes the
prediction disagreement that ordinary dropout tolerates. It does not need a correct guess about
which internal route contains memorization. The old note lists it briefly; the inspected MASSter
loss path has no corresponding two-view consistency term.

**Bounded adaptation.** Generate one candidate set per spectrum. Reuse encoder outputs and run
the intensity head twice with independent token/allocator dropout. Keep context, candidate
membership, valid ion/H channels and targets identical. Both views retain the complete output
set. This tests head consistency first and avoids a second proposal enumeration or encoder
BatchNorm update. Later whole-model consistency is a separate experiment.

Aggregate all candidate ion channels with the same predicted mass into shared peak groups:

`P_v(g) = sum_{(fragment, channel) in group g} P_v(fragment, channel)`.

Use the mean of the two existing supervised losses plus
`lambda * [KL(P_1 || P_2) + KL(P_2 || P_1)] / 2`.

Groups must be formed without target-dependent pruning and must include unmatched predicted
peaks. Use stable log-sum-exp over valid channels; padding/invalid channels do not enter the KL.
Exact isobaric grouping is a clean first definition; instrument-tolerance clustering is a
separate, explicitly recorded approximation.

**Why peak level matters.** Isomeric fragments can explain the same observation. Requiring their
individual latent allocations to agree introduces a constraint the spectrum does not supply.
Likewise, arbitrary candidate row indices are not a common output space across different sets.
Existing `sd1` key hiding is compatible with keeping outputs fixed, but stacking it is a later test.

**Cheap gate.** Measure two-view peak disagreement on a fixed train/val sample first, then screen
small consistency strengths. Compare with two supervised stochastic passes at lambda zero,
not only the one-pass baseline; otherwise extra gradient averaging is a confound. Success requires
better held-out ranking/cosine, not merely smaller KL. A stable lookup can still memorize.

**Gate measured (2026-09-28;
[finding](../findings/masster/masster_dropout_view_disagreement_2026_09_27.md)).**

- Two head passes move 7% (`c0`) to 14% (`k1` family) of training-spectrum intensity: 0.03-0.12
  nats symmetric KL.
- That is nearly as much as on validation. The disagreement follows NCE, not novelty.
- ~~Its validation-side gain at inference equals flattening the deterministic prediction.~~
  That held only for two-view averages. With eight head views the flattened MC-dropout mean beats
  flattened eval on val by +0.0031 on `i1` and +0.0049 on `k1_ema`, at no train cost
  ([follow-up](../findings/masster/masster_dropout_view_disagreement_2026_09_27.md#many-view-follow-up-the-mc-dropout-mean-2026-09-28)).
  So the weight-scaling gap R-Drop targets exists. On `i1` it is below the 0.0042 seed spread.
- `i1` is the control for any arm (user decision, 2026-09-28). Its head noise is allocator
  dropout alone, which `h1` found harmful to raise. Read the arm after the same `P^γ` for arm and
  control, and against `i1` with eight-view head averaging at inference.

**Cost.** One encoder/proposal pass plus two differentiable head passes if implemented this way;
measure runtime and memory rather than assuming a whole-model 2x cost.

## 2. Penalize excessive logit spread: spectral-decoupling-inspired regularization

**Paper.** Pezeshki et al., *Gradient Starvation: A Learning Proclivity in Neural Networks*,
NeurIPS 2021 ([full text](https://arxiv.org/html/2011.09468v3), section 3.4).
The paper analyzes how cross-entropy learning can favor strong features at the expense of weaker
predictive ones, and proposes an L2 logit penalty. Its theory uses a particular classification
setting; it does not prove gradient starvation in MASSter's soft-intensity, marginalized loss.
“Spectral” here refers to the learning analysis, not mass spectra.

**Why here.** This offers a different explanation from a uniquely identifiable memorization
route: some easily fitted features may dominate learning. Weight decay on parameters is not
an output-logit penalty. MASSter is also measurably over-sharp, although post-hoc flattening alone
only improves calibration. This is a low-cost hypothesis worth separating from that effect.

**Bounded adaptation.** Start with valid fragment allocation logits `z_f` before their per-spectrum
softmax. Add `lambda * mean_f((z_f - mean_f(z_f))^2)`, averaged equally over spectra. Centering
removes the unidentifiable common softmax offset; averaging avoids scaling the penalty with the
candidate count. Do not penalize padding or masked sentinel values. Keep ordinary weight decay
unchanged in the first arm so it is a one-change comparison. This is an adaptation, not a faithful
reproduction of the paper's replacement of weight regularization.

This first penalty still depends on the number and identity of fragment explanations. If useful,
check a peak-group version: compute unnormalized channel log weights
`a_fc = z_f + log P(channel | f)`, then `a_g = logsumexp_{fc in g}(a_fc)` and penalize centered
`a_g` instead. Penalizing raw probability logs without accounting for normalization/masks is not
the same operation. A huge penalty would push toward uniform allocation and predictably underfit.

**Gate.** Log logit spread and compare ranking/base-peak accuracy after applying the same
validation-selected post-hoc temperature procedure to both arm and control. If only entropy and
raw cosine improve, count it as calibration rather than newly learned chemistry. Choose a modest
strength using gradient scale, not the possibly negative Gaussian-density CE value.

**Measured premise (2026-09-28;
[finding](../findings/masster/masster_dropout_view_disagreement_2026_09_27.md)): do not build
this form.**

- The centered spread is equal on train and val in `c0`, `k1` and both `k1_ema` weight sets. So
  there is no train-specific extremity for it to remove.
- About half (51-55%) of `mean_f((z_f - mean z)^2)` comes from the ~80% of candidates that hold
  under 1% of predicted intensity. Much of the penalty would go to lifting the zero-intensity tail.
- Inference-time `P^0.875` already gains +0.004-0.005 on val at no train cost.
- A mass-weighted `Var_P(z)` or a peak-level entropy form is the variant worth pricing.

**Related control.** Pereyra et al., *Regularizing Neural Networks by Penalizing Confident Output
Distributions* (2017; [paper](https://arxiv.org/abs/1701.06548)) penalizes low output entropy.
It is distinct from label smoothing and logit L2. For MASSter, define entropy over aggregated
predicted peaks; candidate-uniform regularization silently depends on explanation multiplicity.
Do not launch all three confidence penalties before testing one.

## 3. Short SAM / ASAM continuation, with a plain continuation control

**Papers.** Foret et al., *Sharpness-Aware Minimization for Efficiently Improving Generalization*,
ICLR 2021 ([paper](https://arxiv.org/abs/2010.01412)); Kwon et al., *ASAM*, ICML 2021
([paper](https://proceedings.mlr.press/v139/kwon21b.html)); Zhou et al., *Sharpness-Aware
Minimization Efficiently Selects Flatter Minima Late in Training*, ICLR 2025
([paper](https://arxiv.org/abs/2410.10373)). The last paper reports that short late SAM phases
can approach full SAM training in its studied settings. ASAM adjusts perturbations for parameter
scaling; it is an alternative to test, not guaranteed better here.

**Why here.** EMA and ensembles are the most dependable gains in the record. SAM optimizes
robustness to weight perturbations rather than averaging weights after training. The mechanisms
can overlap but are not identical. The old note already proposes late SAM; the 2025 paper gives
more direct support for a bounded continuation rather than a full expensive run.

**Experiment.** Branch a saved checkpoint into an ordinary optimizer continuation and a SAM
continuation with the same LR schedule, data order, epochs and EMA policy. Initially perturb
allocator and boundary-reader parameters only, leaving the shared encoder/proposal fixed for
this diagnostic. This is partial SAM, not the paper's full-network method. A follow-up may unfreeze
more parameters if held-out gains justify it.

**Engineering constraints.** Reuse the same candidate set and stochastic masks for the two
sharpness evaluations. Updating proposal selection between perturbed and unperturbed passes
changes the discrete support and clouds the local smoothness experiment. If the encoder runs
in training mode, prevent the second pass from updating BatchNorm running statistics again.
Ensure perturbed weights are restored before optimizer/EMA/checkpoint steps. Compare compute as
well as epochs: the differentiable portion needs roughly two passes per update.

*Checked 2026-09-28:*

- `MassterPL` inherits `SpectrumPL`'s automatic-optimization-only `training_step`: manual
  optimization raises `NotImplementedError`. A closure-based SAM step would re-run candidate
  construction and the BatchNorm update.
- The CUDA builder does not return the same candidate set twice even in eval mode. It changes
  fragment order, and about 1% of spectra swap a tied isomer at the budget.
- So fixed candidates need an explicit cache
  ([finding](../findings/masster/masster_dropout_view_disagreement_2026_09_27.md#side-findings)).

*Build review 2026-09-28 (no code changed; numbers from `k1_ema` `j4u25qfd`, epoch 50):*

- **The subset.** On the `k1` lineage, allocator plus boundary reader is `intensity_head` (2.46M)
  plus `boundary_bond_encoder` (0.79M): 35% of 9.21M parameters, LayerNorm only. All BatchNorm
  is in `precursor_encoder`.
- **For this subset, the last two bullets above do not bind.** The perturbation changes nothing
  upstream of the boundary encoder.
  - Split `MassterModel.forward` there. Encode, propose, select and score charge once, so there is
    one candidate set and one BatchNorm update.
  - Run the boundary encoder, allocator and output stages twice, the second under the first
    pass's RNG state so dropout masks match.
  - `autograd.grad` of the first loss with respect to the subset never enters the encoder graph,
    so the second backward reuses it.
  - Lightning 2.4 calls `on_before_optimizer_step` after backward and before clipping and the step
    (`Precision._after_closure`). One automatic-optimization `training_step` can perturb and
    return the perturbed loss; that hook restores the weights. EMA folds in afterwards
    (`on_train_batch_end`).
  - The cache is needed only if the whole forward runs twice. The cost is the split of a
    ~1,300-line forward that has concurrent uncommitted edits. Step overhead is below closure
    SAM's 2x by the encoder/proposal share, not measured.
- **No branch path exists.** The checkpoint holds optimizer moments, EMA training weights, RNG and
  loop state.
  - The runner resumes only its own run: the job-id file maps to the same W&B run, whose saved
    config overwrites the new one.
  - `finetune_pretrained_ckpt` is FraGNNet-only.
  - Two branches need a fresh-run option that hands a foreign checkpoint to
    `trainer.fit(ckpt_path=...)` and rewrites the `ModelCheckpoint` paths. Per-epoch loader seeds
    already give both branches the same data order.
- **Branch point.** The local epoch-50 file is the best-by-val checkpoint. Both arms start at the
  val peak of the split they are read on. The paired contrast stays valid, but a pre-registered
  epoch avoids the selection. Branching 50 → 64 gives a check: the ordinary branch should
  reproduce the original tail (0.7421, epochs 54-63).
- **Radius scale, subset only.** `||w||` = 168; the current Adam step is 0.37 (saved moments,
  lr 1e-3); the EMA average sits 20.7 from the training weights. The textbook ρ = 0.05 is a
  seventh of one step and 1/400 of the distance EMA averages over.
  - Choose ρ from a no-training pilot: `L(w + ε) − L(w)` at a few radii, in eval mode, for the
    training and the EMA weights. Drop tie-swap spectra, as the disagreement probe did.
  - If the EMA weights are already flatter, SAM and EMA should overlap.
- **Noise floor.** The 0.0042 seed spread is for independent full runs. A second ordinary branch
  with another seed measures the branch floor the gate needs.
- **Open:** whether the unperturbed encoder/proposal still train (the control stays the plain
  continuation) or freeze in both arms.

**Gate.** Beat the ordinary continuation at matched epochs and inspect the equal-wall-clock
comparison. An effect within seed variation warrants replication. A null partial/late SAM screen
does not prove that early/full SAM cannot help.

## 4. FLAG / adversarial feature perturbation

**Paper.** Kong et al., *Robust Optimization as Data Augmentation for Large-Scale Graphs*,
CVPR 2022 ([paper](https://openaccess.thecvf.com/content/CVPR2022/html/Kong_Robust_Optimization_As_Data_Augmentation_for_Large-Scale_Graphs_CVPR_2022_paper.html)).
FLAG learns gradient-based perturbations of node features and evaluates graph tasks including
molecular graph classification. This is closer to molecular input than generic image corruption,
but its published result is not a fragmentation-spectrum result.

**Why distinct from failed Gaussian noise.** Random perturbations sample typical directions;
adversarial perturbations target directions that change the prediction most. A null `gm2` does
not establish local robustness. The older note demoted FLAG because the route probe pointed at
global attention; that is not a valid exclusion criterion.

**Adaptation.** Perturb projected continuous atom features or the candidate-token projection,
not atom types, bonds, exact masses, CE or labels. Use a small norm-bound relative to activation
scale and a small number of inner updates. Fixed candidates isolate scoring robustness; token
perturbation is a cheaper initial variant than full graph FLAG. Compare an equal-radius random
noise arm under equal compute. No claim that an embedding perturbation is a physical molecule.

**Gate.** Improved novel-molecule predictions with intact candidate coverage, not merely resistance
to the chosen attack. Too much smoothing can erase real fragmentation cliffs.

## 5. Representation self-challenging

**Paper.** Huang et al., *Self-Challenging Improves Cross-Domain Generalization*, ECCV 2020
([paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3018_ECCV_2020_paper.php)).
RSC suppresses currently dominant features during training to encourage alternative predictive
features. Its empirical evidence is cross-domain image classification.

**MASSter proposal.** Use the spectral objective's gradient to identify a small fraction of
high-influence token feature dimensions, mask those features for a training pass, and keep every
candidate output scored. Refresh the mask with the current model. Do not permanently label a
whole input block as memorizing and remove it. This adapts a classification method to a continuous
spectrum target; it needs its own justification.

**Control and risk.** Compare with random masking at the identical rate; preserve a clean
supervised term in the proposed variant. RSC may simply delete the most useful chemistry.
Because it is another masking method and costs extra differentiation, rank it after R-Drop and
SAM, rather than extrapolating a gain from the paper.

## 6. Reopen distillation as a measured option

**Papers.** Furlanello et al., *Born Again Neural Networks*, ICML 2018
([paper](https://proceedings.mlr.press/v80/furlanello18a.html)), and Hinton et al., *Distilling the
Knowledge in a Neural Network* (2015; [paper](https://arxiv.org/abs/1503.02531)).
The former shows same-architecture students improving over their teachers. The latter explains
using softened teacher distributions to expose information beyond the top prediction.

**Correction to the old ranking.** Near-perfect teacher training fit does not logically exclude
a generalization benefit from self-distillation. Soft-target training changes optimization and
regularization even on the same examples. It can also transfer memorization, so a benefit is
not guaranteed. MASSter's labels are already distributions, which may reduce the additional
information relative to classification with hard labels.

**Proposed gate.** First compare teacher and observed peak distributions after softening, on
training molecules, to establish whether there is useful nontrivial signal. For a student,
retain the observed-spectrum loss and add a teacher term on mass-aligned aggregated peaks;
never align latent candidate row numbers. Use the existing EMA/ensemble teacher and a plain
student with the same budget as control. Treat probability mass outside student support
explicitly; report it rather than silently renormalizing it away.

*Prior (2026-09-28):*

- The closest measured case is `u3`, Switch EMA, which feeds the average back into the training
  weights. It lost −0.0089 against `u2`, with train loss 0.16 higher throughout
  ([ranking-transfer plan](../plans/masster/masster_ranking_transfer_plan.md#switch-ema-arm-u3-built-2026-09-25-not-submitted)).
- A consistency term to the same run's EMA teacher is a soft form of that feedback, so `u3` counts
  against it. It does not bear on out-of-fold teachers.

Out-of-fold teachers are the stronger test of transfer because they never trained on the
molecule they label. They do not require additional molecules, but require additional training.
Use only training folds and keep validation/test out of teacher fitting. This is more expensive
than the first three options, not the next default run.

## 7. Teach cross-molecule geometry explicitly

**Paper.** Zha et al., *Rank-N-Contrast: Learning Continuous Representations for Regression*,
NeurIPS 2023 ([paper](https://papers.nips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html)).
It shapes representation distances according to target-distance ordering rather than relying
on end-to-end regression alone. Its evaluated tasks are not mass spectrometry.

**Why this is a different direction.** Most MASSter arms make memorization harder without
explicitly rewarding transferable geometry. Ordinary instance-contrastive training can push
all different molecules apart, including ones with similar fragmentation. An auxiliary loss
based on measured target relationships would specify what should be shared.

**Exploratory adaptation.** At matched acquisition conditions, construct pairs/triplets of
*different training molecules*. Use measured fragment-spectrum similarity or a deliberately
chosen neutral-loss view to rank target distances, then regularize a small CE-conditioned
projection of the molecular representation to preserve those relations. Retain the full spectral
loss; the auxiliary projection must not bottleneck the entire predictor.

**Do the offline gate first.** Count nontrivial cross-molecule neighbors after controlling
adduct, instrument/acquisition ladder, CE and precursor dominance. If similarity only identifies
precursor mass or all unrelated spectra are equally distant, there is little useful metric
supervision. Neutral-loss similarity is a hypothesis, not a universal chemical invariance;
exclude trivial shared precursor contributions. Reuse train-only targets, not teacher predictions
on validation. Measure target-neighborhood agreement on held-out molecules.

This is cross-*molecule representation* ranking, distinct from another within-spectrum candidate
ranking loss, whose train error is already small. It is also a more speculative adaptation than
R-Drop: no claim that the published RNC guarantees carry over to this distance definition.

## Other papers worth knowing, but lower priority

| Method / source | Relevant idea | Why not the next arm |
| --- | --- | --- |
| [Fishr, ICML 2022](https://proceedings.mlr.press/v162/rame22a.html) | Match gradient variances across training domains. | Scaffold/acquisition domains need careful definition and adequate support; per-example gradients cost time. Random molecule microbatches are not automatically meaningful environments. |
| [Deeply-Supervised Nets, AISTATS 2015](https://proceedings.mlr.press/v38/lee15a.html) | Give intermediate features their own supervised objective. | A local spectral readout could retain gradient after the global path fits; however, this repo already saw harmful shared-head base CE. It needs a genuinely separate readout and a gradient-interference check, not simply restoring base CE. |
| [L2-SP, ICML 2018](https://proceedings.mlr.press/v80/li18a.html) | Regularize toward useful starting weights rather than zero. | An early MASSter checkpoint is not automatically a transferable pretrained representation. Anchor only if its novelty/parts metrics justify it; EMA is already a strong simpler control. |
| [DIR, ICLR 2022](https://arxiv.org/abs/2201.12872), [GREA, KDD 2022](https://arxiv.org/abs/2206.02886) | Separate predictive graph rationale from nuisance environment, then intervene on environment. | Unlike many graph classification settings, the “rest of the molecule” can change charge retention and intensity. Unrestricted environment swapping risks repeating `gm3`–`gm6` with invalid invariance assumptions. |
| [FIORA, Nature Communications 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11889238/) | Learn bond-neighborhood fragmentation behavior and fragment-ion probabilities. | Supports local scoring as a relevant inductive bias, not a guaranteed remedy. MASSter's existing local learner gave only small gains and little at high CE; require the collision/peak-level gate before another local branch. |
| [MassFormer, Nature Machine Intelligence 2024](https://www.nature.com/articles/s42256-024-00816-8) | Global attention with chemical pretraining for spectrum prediction. | Counters the claim that global context itself is necessarily harmful. A different pretrained encoder is a larger intervention, and extra-data pretraining is outside the current scope. |

## Two cheap checks before paying for another training method

1. **BatchNorm recalibration on copied EMA weights.** SWA's original procedure recomputes
   statistics for averaged weights ([authors' paper](https://people.orie.cornell.edu/andrew/swa.pdf),
   section 3.3). Averaging running buffers does not in general yield the moments of activations
   under averaged weights. Recompute on representative training data with frozen parameters,
   controlled batch construction, and dropout disabled; update only BN buffers, then evaluate
   unchanged validation. Compare raw and EMA checkpoints before/after, and repeat the calibration
   sample. The repository's random-batch control was already essentially null on `c0`, so this
   is a cheap EMA-specific check, not a likely explanation for the whole gap.
2. **Frozen-support two-view disagreement.** Quantify how much predicted peak mass moves under
   the current dropout, separately for common/rare chemistry and CE. This prices the R-Drop target
   and distinguishes candidate churn from intensity instability. It is a diagnostic, not a
   guarantee that consistency training helps.
   **Done 2026-09-28**
   ([finding](../findings/masster/masster_dropout_view_disagreement_2026_09_27.md)).
   - Whole-model churn puts only 0.2-0.5% of intensity on one-sided peaks.
   - Disagreement follows NCE; novelty barely moves it.

## Proposed order and stopping rules

- First read already-running arms and finish the matched evaluation described in the review.
  The pending no-attention/mass-window local queue is separate; do not overwrite its configs.
- Cheap checks: EMA BN recalibration and two-view disagreement. *The two-view check ran
  2026-09-28 ([finding](../findings/masster/masster_dropout_view_disagreement_2026_09_27.md)).*
- Low-cost new loss screen: two small centered-logit penalty strengths, with the original CE
  retained. Calibrated ranking decides whether this is more than a temperature substitute.
  *Not as specified: the premise failed; see the §2 measurement.*
- Strongest new training objective: head-only peak R-Drop versus two-pass supervised control.
  *Magnitude gate passed, specificity did not; on `k1` and flattened-calibrated if run (§1).*
- Bounded optimizer test: late partial SAM versus ordinary continuation, with matched schedule
  and an equal-compute read. Test ASAM only if parameter scaling or SAM results justify it.
- Then, if needed: FLAG/token adversarial perturbation; RSC; cross-fitted distillation.
- Research track: train-only target-neighbor census for RNC-style cross-molecule geometry.

Use the same frozen recipe, record seeds and exact checkpoint epochs, keep training and
validation in eval mode for gap measurements, and report molecule-clustered paired uncertainty
separately from seed variation. Retain ranking, novelty/rarity × CE, coverage and wall-clock
metrics. No universal gain threshold or expected cosine improvement is supplied by these papers.

## Verification and limitations

Searched primary publication pages, author manuscripts and official model repositories. Checked
method descriptions and, for spectral decoupling, the proposed penalty in the full text. Several
publisher pages were accessible only through indexed text; this was not a line-by-line audit of
all supplements or code. No benchmark number from another task is treated as an expected MASSter
gain. Cross-checked the older literature note, active route plan, regularization and sampler
findings, and the MASSter head/loss implementation. The checkout contains concurrent model and
CUDA work; this task changes documentation only.
