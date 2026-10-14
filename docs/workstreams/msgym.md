# MassSpecGym (MSGYM) workstream

> **Status:** HUB · **Updated:** 2026-10-14
> MASSter training on MassSpecGym 1.5, formula-track retrieval, GLACIER comparisons, annotation-aware scoring. The ChemBERTa-3 basic model was retired on 2026-10-12.

## Where it stands
- **Sweep read 2026-10-11:** [tiwhun9q results](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#live-results-read-2026-10-11): 15 finished / 33 interrupted; encoder dropout and LR are the strongest signals.
- **Leader re-run ties `pow075` on formula test (2026-10-13):** `f44tzxoc`@19 (val 0.594) gives hit@1 `mspred_ent` 0.523 and `loglik_cal` 0.545; every paired difference against `pow075` crosses zero (`mspred_ent` +0.007 [−0.006, +0.021]); [read](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#leader-re-run-formula-test-retrieval-2026-10-13).
- **Short-schedule / candidate-budget screen PREPARED 2026-10-14:** six arms on the leader recipe at 64 epochs (`c64`, `c32`, `u128`, `e1024`, `u128_e1024`, `floor01`), not submitted ([plan](../plans/masster/msgym_short_budget_screen_2026_10_14.md)).
- **Rare chemistry sampling removed 2026-10-14 (user), never run:** the MSGYM A0/A1/A3 configs and the sampler are in Git at `ae0470d4` ([plan](../plans/masster/rare_chemistry_augmentation_plan.md), [class census](../findings/masster/rare_chemistry_class_census_2026_10_10.md)). The [element-swap census](../findings/masster/rare_chemistry_element_swap_census_2026_10_14.md) found no usable MSGYM swap pairs, so rare-element route 1 is NIST20-only.
- **Best measured MASSter MSGYM result:** `ll04i00m` (GLACIER-parity features arm, epoch 60) reaches test `cos_sim_0.01` 0.5713 against 0.5455 for GLACIER's released checkpoint (+0.026, 99.9% CI +0.009 to +0.044). In sqrt space the gap is +0.011 (CI crosses zero). `1imawkx1` (base recipe) peaked at val 0.5754 (epoch 62) but has no uploaded checkpoint.
- **Retrieval (formula track, simulation-challenge `[M+H]+`, test 9,954 queries):** `ll04i00m` query log-likelihood gives hit@1 0.485 against 0.358 for cosine; the calibrated rule `ll_g0.75_t2_e1e-4_a0.5` gives 0.528. GLACIER is level on ms-pred entropy and below on cosine. Not leaderboard-comparable.
- **Best retrieval checkpoint (2026-10-07):** `pow075` (`q3oxbpgx`@55) reaches `loglik_cal` 0.542 and `mspred_ent` 0.516 (+0.031 over `ll04i00m`, CI excludes zero); `wd10` (`8afoui7e`@50, best val) is within noise of `ll04i00m` on retrieval. Mass track (`pow075` only, no comparator): top-1 `loglik_cal` 74.5, `loglik` 71.4, cosine 52.1 ([arm screen finding](../findings/masster/masster_msgym_arm_screen_formula_retrieval_2026_10_07.md)).
- **Generalization gap is the problem:** `ll04i00m` train 0.884 vs val 0.584; val peaks near epoch 60. No MSGYM seed replicate exists.
- **Screen ran 2026-10-02/03:** control (×3) + `s2` + arms, 18 checkpoints in W&B; val best spans 0.571–0.587 against a control spread of 0.0025 ([finding](../findings/masster/masster_msgym_arm_screen_formula_retrieval_2026_10_07.md)). The four calibrated-likelihood arms (`pow075`, `pm1`, `sqrt_pm1`, `pow075_pm1`) ran in that screen. Still prepared, not submitted (2026-10-08): `pow075_s2` seed replicate, `b1024` arm, a `b256` rerun (`1p4s2w61` crashed at epoch 110, 3.94 h runtime, 25.7 GB peak, so not OOM; likely the wall-time limit, unconfirmed until the Fir job log is read), five U256/E1024 configs, ; sweep `tiwhun9q` has now run (see the live read below), using pow0.75 at `loss_batch_size` 16 ([training-arms plan](../plans/masster/masster_msgym_training_arms_plan.md), [sweep plan](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md)). The retrieval runner has run locally (formula test, three checkpoints), never on a cluster.
- **`loss_batch_size` is 16 in every `config/masster/massspecgym/` config since 2026-10-08** (was 64 in 25 of them). It only sets the padded loss-chunk window (accumulation order, not a modeling knob; [slice sizing](../plans/masster/masster_regularization_sweep_plan.md#slice-sizing-is-a-measured-open-question-not-a-settled-choice)) and lets batch-512 runs fit `fir:h100-3g.40gb`. Every run up to and including the 2026-10-03 screen trained at 64; W&B hparams, not the config files, record that.
- **Branch `rare-chemistry-sampling`:** no MSGYM code or configs remain there.
- **Single next action:** re-select `loglik_cal` on val for `pow075` and `wd10`; a second `pow075` seed to check the entropy/likelihood gain.

## Open threads
| Thread | Status | Next concrete step |
|---|---|---|
| [Rare chemistry augmentation](../plans/masster/rare_chemistry_augmentation_plan.md) | PREPARED | MSGYM arms removed 2026-10-14; route 1 is NIST20-only |
| [MSGYM training arms](../plans/masster/masster_msgym_training_arms_plan.md) | screen RAN 2026-10-03 | Val read in the [arm screen finding](../findings/masster/masster_msgym_arm_screen_formula_retrieval_2026_10_07.md); test retrieval done for `wd10`, `pow075`; other arms not retrieval-tested. |
| [Short-schedule / budget screen](../plans/masster/msgym_short_budget_screen_2026_10_14.md) | PREPARED | Submit `submit_msgym_short_budget.sh all` on Fir; check `e1024`/`floor01` first epoch. Supersedes the batch-512 U256/E1024 configs below for the budget question. |
| [U256/E1024 follow-up](../plans/masster/masster_fragment_marginal_selection_plan.md#msgym-u256e1024-follow-up-2026-10-05-prepared-not-run) | PREPARED | Real-data GPU smoke; record launch provenance; matched comparison with U256/E4096. |
| [48-trial val-loss sweep](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md) | RAN; leader re-run retrieval-tested | Leader `f44tzxoc` ties `pow075` on formula test. Next: matched control or a seed replicate before any further retrieval run. |
| [Annotation-aware retrieval](../plans/annotation_aware_retrieval_plan.md) | ACTIVE | `[M+Na]+` and no-CE policy for full-challenge numbers; checkpoint ensemble on val; Gate 1 baseline fixed by the user. |
| [Retrieval runner](../guides/run_msgym_retrieval.md) | built, not run on a cluster | Submit a test-fold job for a new checkpoint after re-selecting `loglik_cal` on val. |
| GLACIER CF NIST'20 shortcut | argument only | Needs MAGMa labelling and a retrain; judged not worth it ([swap test](../findings/masster/glacier_cf_swap_test_2026_10_03.md)). |

## Settled — read before re-proposing
**MASSter vs GLACIER**
- [Released-checkpoint comparison](../findings/masster/masster_msgym_vs_glacier_released_ckpt_2026_09_24.md) — MASSter +0.026 on linear cosine; GLACIER outputs sqrt intensity (square it); no established QTOF winner.
- [Formula retrieval, full](../findings/masster/masster_msgym_formula_retrieval_full_2026_10_03.md) — test hit@1 cos 0.358 / ll 0.485 / calibrated 0.528; GLACIER level on entropy via sub-1% peaks; reproduces paper's w/ CF row by entropy.
- [GLACIER CF swap test](../findings/masster/glacier_cf_swap_test_2026_10_03.md) — wrong-formula spectrum ranks truth at chance for both; same-formula wrong spectrum above chance, MASSter higher. No CF structure prior.
- [GLACIER gap = CE allocation (NIST)](../findings/fragnnet/glacier_gap_ce_decomposition_2026_09_21.md) — FraGNNet-side gap sits at high CE; coverage not binding.

**Scoring and retrieval**
- [Arm screen + formula retrieval](../findings/masster/masster_msgym_arm_screen_formula_retrieval_2026_10_07.md) — `pow075` +0.031 `mspred_ent` / +0.018 `loglik` hit@1 over `ll04i00m`; `wd10` within noise; val cosine does not rank checkpoints for retrieval.
- [Annotation pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md) — likelihood +14.8 hit@1 over cosine, mostly small-peak weighting; fragment cost adds nothing to the likelihood; support worse than cosine.
- Retrieval pipeline facts: truth is candidate 0 (never tie-break by position); identify truth by MSGYM source key; standardizing candidates changes no score (±0.005).
- Entropy is how the GLACIER paper ranks retrieval; ms-pred bootstrap treats spectra as independent, so its intervals are too narrow.

**Training**
- GLACIER-parity features arm: no gain (0.5669 vs 0.5695 tail); both MSGYM recipes peak near epoch 60 and drift down ([arms plan](../plans/masster/masster_msgym_training_arms_plan.md)).
- User scope decision 2026-09-24: benchmark subset only, no extra MassSpecGym data (`ms_gym_all_1_5` would add 9,313 molecules).
- [Marginal selection](../plans/masster/masster_fragment_marginal_selection_plan.md) — CLOSED NULL on NIST; U128 ties U256 at 1.65x speed.
- [ChemBERTa-3 E1/T2](../plans/chemberta3_msgym_plan.md#results-2026-09-28-runs-finished) — peak after epoch 0 at val 0.199/0.194, end 0.169/0.152; structure-blind prior scores 0.194; random SMILES hurt. `chemlm` retired 2026-10-12 (E4 never ran); code at `13b1599d`.

## All documents
| Doc | Status (updated) | One-line content |
|---|---|---|
| [Rare chemistry augmentation](../plans/masster/rare_chemistry_augmentation_plan.md) | PREPARED (2026-10-14) | Shared NIST20/MSGYM pilot; sampling arms removed 2026-10-14, NIST20 route 1 prepared |
| [rare_chemistry_class_census](../findings/masster/rare_chemistry_class_census_2026_10_10.md) | FINDING (2026-10-10) | Class-rarity census; A1 exposure shift, A3 fails support gate. |
| [masster_msgym_training_arms_plan](../plans/masster/masster_msgym_training_arms_plan.md) | PREPARED (2026-10-06) | Control, 10-arm screen, sqrt, calibrated arms, baseline history, scope decision, ms-pred CE handling. |
| [msgym_short_budget_screen](../plans/masster/msgym_short_budget_screen_2026_10_14.md) | PREPARED (2026-10-14) | 64/32-epoch controls and U128, E1024, fragment-floor arms on the sweep-leader recipe. |
| [msgym_val_loss_sweep_48](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md) | ACTIVE (2026-10-11) | Live 48-trial read, interruption audit, hyperparameter signals and checkpoint recovery. |
| [masster_fragment_marginal_selection_plan](../plans/masster/masster_fragment_marginal_selection_plan.md) | CLOSED NULL (2026-10-06) | NIST marginal-selection null; holds the MSGYM U256/E1024 config list. |
| [annotation_aware_retrieval_plan](../plans/annotation_aware_retrieval_plan.md) | ACTIVE (2026-10-06) | Retrieval scoring plan, frozen full-track protocol, calibration. |
| [chemberta3_msgym_plan](../plans/chemberta3_msgym_plan.md) | CLOSED NULL (2026-10-12) | ChemBERTa-3 first 6 layers end to end with NEIMS head; results; model retired. |
| [masster_msgym_vs_glacier_released_ckpt](../findings/masster/masster_msgym_vs_glacier_released_ckpt_2026_09_24.md) | FINDING (2026-10-06) | Test cosine comparison, peak over-prediction, instrument audit. |
| [masster_msgym_arm_screen_formula_retrieval](../findings/masster/masster_msgym_arm_screen_formula_retrieval_2026_10_07.md) | FINDING (2026-10-07) | Val table of all 18 screen checkpoints; formula test retrieval for `wd10`, `pow075`, `ll04i00m` with paired CIs. |
| [masster_msgym_formula_retrieval_full](../findings/masster/masster_msgym_formula_retrieval_full_2026_10_03.md) | FINDING (2026-10-06) | Full val+test formula retrieval, GLACIER matching, calibrated likelihood. |
| [masster_annotation_retrieval_pilot](../findings/masster/masster_annotation_retrieval_pilot_2026_10_02.md) | FINDING (2026-10-06) | 400-query pilot: likelihood, fragment cost, mass track, GLACIER. |
| [glacier_cf_swap_test](../findings/masster/glacier_cf_swap_test_2026_10_03.md) | FINDING (2026-10-03) | Wrong-spectrum swap test of GLACIER contrastive fine-tuning. |
| [run_msgym_retrieval](../guides/run_msgym_retrieval.md) | GUIDE (2026-10-04) | One-command prepare/infer/score/summarize runner. |
| [glacier_object_detection](../notes/glacier_object_detection.md) | NOTE (2026-10-02) | GLACIER paper note (CF, ms-pred datasets). |
| [massspecgym_in_the_wild_audit](../notes/massspecgym_in_the_wild_audit.md) | NOTE (2026-09-20) | Evaluation audit of MassSpecGym. |
| [aime_deepms2reasoner](../notes/aime_deepms2reasoner.md) | NOTE (2026-09-28) | AIMe / DeepMS2Reasoner; its 0.83 is on formula-filtered targets. |
| [molspecflow](../notes/molspecflow.md) | NOTE (2026-09-20) | MolSpecFlow paper note. |
| [glacier_gap_ce_decomposition](../findings/fragnnet/glacier_gap_ce_decomposition_2026_09_21.md) | owned by [FraGNNet hub](fragnnet.md) | GLACIER gap on NIST20 is high-CE allocation. |

Neighbours: [MASSter core](masster_core.md) · [MASSter generalization](masster_generalization.md) · [Spectraverse](spectraverse.md) · [NIST23 EI](nist23_ei.md) · [CASMI / Enveda](casmi_enveda.md) · [FraGNNet](fragnnet.md).

## Key artifacts
- Configs: `config/masster/massspecgym/` (control `…_cuda_k1_ema999_fourier.yml`; arms `…_<arm>.yml`; `msgym_u256_e1024_*.yml`; `msgym_val_loss_sweep_40gb.yml`).
- Sweep: `sweep/masster/msgym_val_loss_48_40gb.yml`, W&B [tiwhun9q](https://wandb.ai/frag-gnn/frag-gnn/sweeps/tiwhun9q); launcher `scripts/manual_slurm/fir/prepare_msgym_val_loss_sweep.sh`.
- Runs: `ll04i00m` (checkpoint uploaded), `1imawkx1` (no checkpoint), ChemBERTa E1 `tm9qwq78`, T2 `ku742mhq`.
- Retrieval code: `scripts/ms2c/run_msgym_retrieval.py`, `scripts/manual_slurm/prepare_msgym_retrieval.sh`, `scripts/eval/score_retrieval_candidates.py`, `scripts/eval/summarize_retrieval.py`, `src/fragnnet/utils/retrieval_utils.py`.
- Retrieval outputs: `output/msgym_formula_retrieval_2026_10_02/`, `output/glacier_msg_eval_2026_09_24/`, pilot `output/annotation_retrieval_pilot_2026_10_02/`.
