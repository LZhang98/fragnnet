# MASSter generalization on NIST20 (the `i1` line)

> **Status:** HUB · **Updated:** 2026-10-10
> Why MASSter fits training molecules and ranks new ones poorly on unmerged NIST20 `[M+H]+`: the memorization routes, `i1` error analyses, information gates and the SMARTS/excision/isotope/JEPA side threads.

## Where it stands

- **Prepared, not submitted:** [rare chemistry sampling](../plans/masster/rare_chemistry_augmentation_plan.md) A0/A1/A3 on `i1` (`config/masster/rare_chemistry_2026_10_10/`, branch `rare-chemistry-sampling`). [Rare chemistry census](../findings/masster/rare_chemistry_class_census_2026_10_10.md): A1 shifts molecule exposure by total variation 0.109; A3 barely differs from A1 and fails the support gate, so submit A0 and A1 unless the user redesigns A3. Rare elements: [revised plan](../plans/masster/rare_chemistry_augmentation_plan.md#rare-element-handling-revised-2026-10-14) checks the `i1` gap first, then a Br↔Cl transfer screen ([element-swap census](../findings/masster/rare_chemistry_element_swap_census_2026_10_14.md)).
- **Best recipe:** `i4` = `i1` (instrument input on `ORB`/`ORB_F`) + `k1` dropout 0.2 + weight EMA 0.999 + intensity pre-norm. Tail mean of `val_cos_sim_0.01_epoch/mean`, epochs 54-63: **0.7729** (W&B `rd9rsdqx`), +0.0019 over `i3` (`xypkkfti`, 0.7710) and +0.0102 / +0.0122 over `i1` (`i50ig7rf`, 0.7607). Seed spread 0.0042; single-seed gate +0.005. Configs: `config/masster/mh_ft_memorization_route_2026_09_24/{i1,i3,i4}_*_ob_obf.yml`.
- **Open problem:** the validation floor, not the average. 330 spectra / 119 molecules score below 0.05 at instrument tolerance (train sample: 4); 88% are ranking failures with the candidates present. No tested arm (`fa1`, `i3`, cut probe, peak-ranking loss, fragment connectivity) moves a floor chosen by an independent checkpoint.
- **Single next action:** the computed-energetics information gate. Formal floor coverage is 97% but only 14 / 24 harder channels keep a distance-preserved product pair, so first fix a bounded, target-independent state/calculation policy and a train/calibration/confirmation protocol ([plan](../plans/masster/masster_memorization_route_plan.md#computed-product-ion-energetics-information-gate-2026-10-01-proposed)).
- **Prepared, not submitted:** `so1` (`i1` + SMARTS templates, aryl-sulfonamide O transfer; Fir stage `smarts`), `fc1`/`fc1p` (configs exist, not in the submitter; local screen not positive).
- **Branch `rare-chemistry-sampling`:** rare chemistry sampler, censuses, configs and plan are committed there, not on main; JEPA code and data removal is committed (`6f2fdea2`).
- Core architecture/training lives in [masster_core](masster_core.md); retrieval in [msgym](msgym.md); other datasets in [spectraverse](spectraverse.md), [nist23_ei](nist23_ei.md), [casmi_enveda](casmi_enveda.md); the sibling model in [fragnnet](fragnnet.md).

## Open threads

| Thread | Status | Next concrete step |
| --- | --- | --- |
| [Rare chemistry augmentation](../plans/masster/rare_chemistry_augmentation_plan.md) | PREPARED | A0/A1 and rare-element route 1 (E0/E1/E2: `a_periodic`, hidden-Br test) ready to submit; A3 needs a user redesign decision |
| [Memorization-route plan](../plans/masster/masster_memorization_route_plan.md) (dated log: [history](../history/masster_memorization_route_plan_log.md)) | ACTIVE | Energetics gate protocol (above); coverage run `so1` when the user decides |
| [Energetics feasibility](../findings/masster/masster_energetics_feasibility_2026_10_02.md) | FINDING | State/calculation policy, missing-feature handling, independent selector, effect size, before any fit |
| [Transfer-oriented (MLDG-style) training](../plans/masster/masster_memorization_route_plan.md#transfer-oriented-training-for-covered-ranking-failures-2026-10-01-deferred) | DEFERRED | Only a cheap head-gradient conflict check on training-only batches, if revived |
| [Intensity displacement](../plans/masster/masster_intensity_displacement_plan.md) | DEFERRED | Re-derive premises on `i1` first; one cheap hashed-environment retry of the bond-local learner is open |
| [Ranking transfer](../plans/masster/masster_ranking_transfer_plan.md) | DEFERRED | EMA already carried into `i3`/`i4`; gates 2-3 unread, no step pursued |
| [Intensity reliability](../plans/masster/masster_intensity_reliability_plan.md) | DEFERRED | P2-P5 not run; needs replicate provenance upstream |
| [Experiment reopening audit](../findings/experiment_reopening_audit_2026_09_28.md) | FINDING | Untested alternatives: out-of-fold ensemble distillation, narrower R-Drop and smooth-CE follow-ups |
| [Dropout-view disagreement](../findings/masster/masster_dropout_view_disagreement_2026_09_27.md) | FINDING | User choice: R-Drop arm pair on `i1`, or 8-view head averaging at inference (+0.0031, no training) |

## Settled — read before re-proposing

**User decisions**

- [`f0`/`f1` retired](../findings/masster/masster_memorization_attempts_audit_2026_09_27.md) — 11 findings on old `FT` labels are SUPERSEDED (rows below); cite the `i1` reads, never `f0`/`f1`/`c0` numbers as current.
- [Worst predictions](../findings/masster/masster_i1_worst_predictions_2026_09_30.md) — rank by `cos_hun_inst_tol` (instrument tolerance), not 0.01 Da bins; 497 of 903 "worst" logged spectra score >= 0.5.
- [Memorization routes](../plans/masster/masster_memorization_route_plan.md) — try many routes, never a single-input block (identity reaches every head input); seed spread 0.0042; `i1` is the control for new arms.
- [SMARTS goal](../findings/masster/masster_smarts_outlier_reach_2026_09_29.md) — raise the lower bound (outlier spectra); NIST20 only, no MONA; 8 rules commented out, re-check on Enveda-180.
- [JEPA removal](../plans/masster/masster_jepa_embedding_plan.md) — code, data and `dreams` env removed 2026-10-04 (user); other null features deliberately kept.
- [Cross-molecule code](../plans/masster/masster_cross_molecule_supervision_gate.md) — `cm0`-`cm2` reverted at the user's request; census and artifacts kept; never run.

**The train/val gap and what moved it**

- [`i1` memorization read](../findings/masster/masster_i1_memorization_read_2026_09_28.md) — instrument input narrows the eval gap 0.183 to 0.158; about 86% remains; Lumos bias fixed.
- [Attempts audit](../findings/masster/masster_memorization_attempts_audit_2026_09_27.md) — read first for any memorization arm; +0.003 gate is about one standard error.
- [Plan review](../findings/masster/masster_memorization_plan_review_2026_09_27.md) — separate held-out promotion from mechanistic criteria; route corruption measures dependence only.
- [Route plan, current summary](../plans/masster/masster_memorization_route_plan.md#current-summary) — `i3` +0.0102 and `i4` pass; full attention +0.0044 just under the gate; route removal, noise, masks, windows and m/z/loss positions all null or worse.
- [Literature](../notes/memorization_literature_2026_09_24.md) and [methods note](../notes/masster_generalization_methods_2026_09_27.md) — ranked list; decision C constant LR; R-Drop/SAM gates.
- [Dropout-view disagreement](../findings/masster/masster_dropout_view_disagreement_2026_09_27.md) — 8-view mean beats best flattened eval by +0.0031 on `i1`; centered-logit penalty not buildable as specified.

**`i1` error analyses (no missing input, hedging, transfer failure)**

- [Base-peak table](../findings/masster/masster_i1_base_peak_errors_2026_09_28.md) — base peak missed in 40% of val (train 15%); pair biases are hedging; no single missing distinction.
- [Residual structure](../findings/masster/masster_i1_residual_structure_2026_09_28.md) — no recorded metadata left; per-molecule energy scale +0.010 val but unpredictable; read before any new conditioning input.
- [Post-cut gate](../findings/masster/masster_i1_postcut_information_gate_2026_09_29.md) — connectivity/charge descriptors add <= +0.001 over the `P^gamma` baseline.
- [`fc1` screen](../findings/masster/masster_fragment_connectivity_local_screen_2026_09_29.md) — +0.0050 mean, below the 0.0076 control spread; no Fir run.
- [Fragment pairs](../findings/masster/masster_i1_fragment_pairs_2026_09_30.md) and [pair contrasts](../findings/masster/masster_i1_pair_contrasts_2026_09_30.md) — 73% of ranking pairs have several compatible atom sets; only familiarity replicates, no cut rule.
- [Isomer split Step 0](../findings/masster/masster_isomer_split_step0_2026_10_01.md) — `i1`/`fa1` agree on 85.9% of confident splits; cut-class prior adds nothing; no arm built.
- [Metadata usage audit](../findings/masster/masster_metadata_usage_audit_2026_10_03.md) — adduct and MSGYM fragmentation-mode embeddings are constants; compared with MassFormer/ICEBERG/FIORA/GLACIER.

**Closed information gates and probes (NIST20, frozen `i1`)**

- [Cut-score probe](../findings/masster/masster_cut_intensity_probe_2026_10_01.md) — additive boundary-score correction selects zero; closes this form, not every cut-supervision design.
- [Peak-ranking pilot](../findings/masster/masster_peak_ranking_pilot_2026_10_02.md) — tuning selects zero ranking weight; full-data end-to-end untested.
- [Structural alignment](../findings/masster/masster_structural_alignment_audit_2026_09_29.md) — exact connectivity keeps 0.85% of measured intensity on unfamiliar queries; stop.
- [Reference information gate](../findings/masster/masster_reference_information_gate_2026_09_29.md) — ion corrections select zero; loss alignment lowers cosine (0.6287 vs 0.6385).
- [Candidate correspondence](../findings/masster/masster_candidate_correspondence_2026_09_29.md) — coverage advantage does not predict accuracy gain; [transfer plan](../plans/masster/masster_candidate_transfer_plan.md) STOPPED.
- [Cross-molecule gate](../findings/masster/masster_cross_molecule_supervision_gate_2026_09_29.md) — soft relationship signal +0.050 at Tanimoto <= 0.4; hard shared-winner labels unsupported; never trained.
- [Local context allocation](../plans/masster/masster_local_context_allocation_plan.md) — frozen-token probe stopped at A1 epoch 17; local term <= +0.001.

**QET / NCE input and instruments**

- [Breakdown-curve form fit](../findings/breakdown_curve_form_fit_2026_09_26.md) — Arrhenius form fits the CE series (0.995); no survival head.
- [QET head gates and NCE input](../findings/masster/masster_qet_projection_candidate_union_2026_09_26.md) — smooth CE inputs fail; `n4` fourier +0.0053 over `c0`.
- [Orbitrap CE scales](../findings/nist_orbitrap_instrument_ce_scales_2026_09_28.md) — two ladders are Elite vs Lumos; `ORB`/`ORB_F` from raw MSP; `FT` and `OB`/`OB_F` refused. `ce1`/`ce2` (NCE into the encoder) +0.0003/+0.0005, no pass.

**SMARTS, excision, isotopes**

- [SMARTS pathway](../plans/masster/masster_smarts_pathway_plan.md) / [gates](../findings/masster/masster_smarts_pathway_gates_2026_09_30.md) — CLOSED NULL: G0 +0.00125 < 0.003, G1 STOP, Track B 0 promoted; `p1` code stays built, off.
- [SMARTS CUDA audit](../findings/masster/masster_smarts_cuda_audit_2026_09_27.md) — templates OFF by default; match standardized molecules; SMARTS vs CUDA-template mismatch is a hard error.
- [SMARTS on Enveda-180](../findings/masster/masster_smarts_enveda_gate_2026_09_29.md) — only S=O rules pass, 2-10x weaker than NIST20; none of the 8 disabled rules finds support.
- [Excision](../findings/masster/masster_excision_continuation_gate_2026_09_28.md) — only the aromatic nitro NO-loss family confirms (+0.648% class intensity); no channel built.
- [Isotope envelope](../plans/masster/masster_isotope_envelope_plan.md) — CLOSED NULL: perfect monoisotopic + best scalar envelope recovers 0.0047 cosine against a 0.005 gate.
- [JEPA embedding](../plans/masster/masster_jepa_embedding_plan.md) — CLOSED NULL: `j1a` +0.0000, `j1b` -0.0003, `j2` -0.0031 against `i1`.

## All documents

| Doc | Status (updated) | One-line content |
| --- | --- | --- |
| [Rare chemistry augmentation](../plans/masster/rare_chemistry_augmentation_plan.md) | PREPARED (2026-10-14) | Shared NIST20/MSGYM rarity census, capped sampling, conditional replicate augmentation and confirmation gates; A1/A3 sampler built |
| [Memorization-route plan](../plans/masster/masster_memorization_route_plan.md) | ACTIVE (2026-10-06) | Route-removal arm queue, gates, current summary; open energetics and MLDG designs |
| [Memorization-route log](../history/masster_memorization_route_plan_log.md) | REFERENCE (2026-10-06) | Verbatim dated log of every arm, read and build (batches 1-5, `i1` stack, ladders, `ce1`/`ce2`) |
| [Intensity displacement plan](../plans/masster/masster_intensity_displacement_plan.md) | DEFERRED (2026-10-06) | Phase 0 / 1.4b on retired `f0`/`f1`; arm 2A not submitted |
| [Ranking-transfer plan](../plans/masster/masster_ranking_transfer_plan.md) | DEFERRED (2026-10-06) | `u2` EMA, `h1`, `h2`, `u3` arms; EMA carried into `i3` |
| [Intensity reliability plan](../plans/masster/masster_intensity_reliability_plan.md) | DEFERRED (2026-10-06) | Ladder-aware replicate noise, P1 census, uncertainty ensembles |
| [Local context allocation plan](../plans/masster/masster_local_context_allocation_plan.md) | STOPPED (2026-10-06) | Constrained context correction; Stage 1 probe stopped by the user |
| [Candidate transfer plan](../plans/masster/masster_candidate_transfer_plan.md) | STOPPED (2026-09-29) | Candidate correspondence and reference-intensity transfer; Gate B negative |
| [Cross-molecule supervision gate plan](../plans/masster/masster_cross_molecule_supervision_gate.md) | STOPPED (2026-09-30) | cm0-2 census-gated training pilot; code reverted |
| [Cut-intensity probe plan](../plans/masster/masster_cut_intensity_probe.md) | CLOSED NULL (2026-10-01) | Frozen cut-to-intensity probe protocol |
| [Peak-ranking pilot plan](../plans/masster/masster_peak_ranking_pilot_2026_10_02.md) | CLOSED NULL (2026-10-02) | CPU intensity-head continuation protocol |
| [Excision channel probe plan](../plans/masster/masster_excision_channel_probe.md) | FINDING (2026-10-06) | Internal-bridge excision diagnostic protocol (completed) |
| [SMARTS pathway plan](../plans/masster/masster_smarts_pathway_plan.md) | CLOSED NULL (2026-09-30) | Pre-registered G0/G1/G2 gates for SMARTS in the pathway |
| [Isotope envelope plan](../plans/masster/masster_isotope_envelope_plan.md) | CLOSED NULL (2026-10-03) | Ceiling for modelling isotope satellites on Enveda/NIST20 |
| [JEPA embedding plan](../plans/masster/masster_jepa_embedding_plan.md) | CLOSED NULL (2026-10-04) | DreaMS-embedding auxiliary; Gate 1 fails; code removed |
| [Bond-local learner](../findings/masster/masster_bond_local_learner_2026_09_24.md) | SUPERSEDED (2026-09-30) | `f0`/`f1`-era identity-blind learner; +0.002 |
| [Candidate correspondence](../findings/masster/masster_candidate_correspondence_2026_09_29.md) | FINDING (2026-09-29) | Endpoint and NCE-band coverage audits |
| [Cross-molecule gate](../findings/masster/masster_cross_molecule_supervision_gate_2026_09_29.md) | FINDING (2026-09-29) | Training-only census of transferable spectral relationships |
| [Rare chemistry census](../findings/masster/rare_chemistry_class_census_2026_10_10.md) | FINDING (2026-10-10) | Shared NIST20/MSGYM class-rarity census; A1 exposure shift, A3 fails support gate |
| [Element-swap census](../findings/masster/rare_chemistry_element_swap_census_2026_10_14.md) | FINDING (2026-10-14) | Measured same-group swap pairs: 57 Br→Cl training pairs on NIST20; none usable for Se/As/Si or on MSGYM |
| [Cut-intensity probe](../findings/masster/masster_cut_intensity_probe_2026_10_01.md) | CLOSED NULL (2026-10-01) | Additive cut-score correction selects zero |
| [Dropout-view disagreement](../findings/masster/masster_dropout_view_disagreement_2026_09_27.md) | FINDING (2026-09-28) | R-Drop and logit-penalty pricing; 8-view averaging |
| [Energetics feasibility](../findings/masster/masster_energetics_feasibility_2026_10_02.md) | FINDING (2026-10-02) | xTB feasibility, joint-state coverage 97%, geometry limits |
| [Excision continuation gate](../findings/masster/masster_excision_continuation_gate_2026_09_28.md) | FINDING (2026-09-28) | Nitro NO-loss family confirms; bridge excision unsupported |
| [Formula-only baseline](../findings/masster/masster_formula_only_baseline_2026_09_23.md) | SUPERSEDED (2026-09-30) | How much cosine formula statistics explain (`f1`) |
| [Fragment-connectivity screen](../findings/masster/masster_fragment_connectivity_local_screen_2026_09_29.md) | FINDING (2026-09-29) | `fc1` local screen, not positive |
| [`i1` base-peak errors](../findings/masster/masster_i1_base_peak_errors_2026_09_28.md) | FINDING (2026-09-28) | Step 1: wrong base-peak table, hedging biases |
| [`i1` fragment pairs](../findings/masster/masster_i1_fragment_pairs_2026_09_30.md) | FINDING (2026-09-30) | Ambiguity-preserving pair table, 52,020 val spectra |
| [`i1` memorization read](../findings/masster/masster_i1_memorization_read_2026_09_28.md) | FINDING (2026-09-28) | Route-map probe on `i1`; gap 0.158 |
| [`i1` pair contrasts](../findings/masster/masster_i1_pair_contrasts_2026_09_30.md) | FINDING (2026-09-30) | Step 2: matched success/failure contrasts |
| [`i1` post-cut gate](../findings/masster/masster_i1_postcut_information_gate_2026_09_29.md) | FINDING (2026-09-29) | Step 2: post-cut/charge descriptors add nothing |
| [`i1` residual structure](../findings/masster/masster_i1_residual_structure_2026_09_28.md) | FINDING (2026-09-28) | Where the val deficit is; no missing metadata |
| [`i1` worst predictions](../findings/masster/masster_i1_worst_predictions_2026_09_30.md) | FINDING (2026-10-06) | Floor of 330 spectra; ranking failure; `so1` arm |
| [Displacement Phase 0](../findings/masster/masster_intensity_displacement_phase0_2026_09_24.md) | SUPERSEDED (2026-09-30) | Where misplaced intensity goes (`f0`/`f1`) |
| [Isomer split Step 0](../findings/masster/masster_isomer_split_step0_2026_10_01.md) | FINDING (2026-10-01) | Within-peak isomer split gate |
| [Matched intensity filter](../findings/masster/masster_matched_intensity_filter_2026_09_24.md) | SUPERSEDED (2026-09-30) | Target cleanup on `3bej6fq7` |
| [Memorization attempts audit](../findings/masster/masster_memorization_attempts_audit_2026_09_27.md) | FINDING (2026-09-27) | Every arm aimed at the gap, re-derived from code and W&B |
| [Energy vs parts](../findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md) | SUPERSEDED (2026-09-30) | Is the lookup keyed on energy, molecules or parts |
| [Memorization plan review](../findings/masster/masster_memorization_plan_review_2026_09_27.md) | FINDING (2026-09-27) | Review of diagnosis and next experiments |
| [Route probe](../findings/masster/masster_memorization_route_probe_2026_09_23.md) | SUPERSEDED (2026-09-30) | Which head input carries memorization |
| [Metadata usage audit](../findings/masster/masster_metadata_usage_audit_2026_10_03.md) | FINDING (2026-10-03) | CE/adduct/instrument/mode vs published models |
| [Peak-ranking pilot](../findings/masster/masster_peak_ranking_pilot_2026_10_02.md) | CLOSED NULL (2026-10-02) | Explicit ranking loss gives no gain |
| [Per-peak uncertainty](../findings/masster/masster_per_peak_uncertainty_2026_09_24.md) | SUPERSEDED (2026-09-30) | Ensemble spread as uncertainty (`f0`/`f1`) |
| [QET projection / candidate union](../findings/masster/masster_qet_projection_candidate_union_2026_09_26.md) | FINDING (2026-09-28) | QET head gates; NCE input; `n4` fourier |
| [Ranking decomposition](../findings/masster/masster_ranking_decomposition_2026_09_23.md) | SUPERSEDED (2026-09-30) | Ranking vs coverage vs calibration (`f0`) |
| [Reference information gate](../findings/masster/masster_reference_information_gate_2026_09_29.md) | CLOSED NULL (2026-09-29) | Aligned reference intensities do not help `i1` |
| [Route map](../findings/masster/masster_route_map_2026_09_24.md) | SUPERSEDED (2026-09-30) | Where the lookup is written (`f0`); `i1` version in the memorization read |
| [SMARTS CUDA audit](../findings/masster/masster_smarts_cuda_audit_2026_09_27.md) | FINDING (2026-09-28) | Templates on CUDA; proposal reach |
| [SMARTS on Enveda-180](../findings/masster/masster_smarts_enveda_gate_2026_09_29.md) | FINDING (2026-09-29) | Per-class coverage gate on all 41 rules |
| [SMARTS outlier reach](../findings/masster/masster_smarts_outlier_reach_2026_09_29.md) | FINDING (2026-09-29) | Rules matter on 37 outlier val spectra |
| [SMARTS pathway gates](../findings/masster/masster_smarts_pathway_gates_2026_09_30.md) | FINDING (2026-09-30) | G0, G1, G2, G2b all fail |
| [Structural alignment audit](../findings/masster/masster_structural_alignment_audit_2026_09_29.md) | CLOSED NULL (2026-09-29) | Exact connectivity retains little intensity |
| [Target cleaning cross-eval](../findings/masster/masster_target_cleaning_cross_eval_2026_09_23.md) | SUPERSEDED (2026-09-30) | Cleaned `[M+H]+`/FT target (was CLOSED NULL) |
| [Variance vs novelty](../findings/masster/masster_variance_novelty_2026_09_24.md) | SUPERSEDED (2026-09-30) | Is run-to-run variance epistemic (`f0`/`f1`) |
| [Breakdown-curve form fit](../findings/breakdown_curve_form_fit_2026_09_26.md) | FINDING (2026-09-26) | Two-parameter QET form on NIST20 CE series |
| [Orbitrap CE scales](../findings/nist_orbitrap_instrument_ce_scales_2026_09_28.md) | FINDING (2026-10-01) | Elite vs Lumos; `ORB`/`ORB_F`; `i1`/`i2` |
| [Experiment reopening audit](../findings/experiment_reopening_audit_2026_09_28.md) | FINDING (2026-09-28) | 125 records screened for directions worth reopening |
| [SMARTS usage audit](../findings/smarts_usage_audit_2026_09_23.md) | FINDING (2026-09-23) | Every SMARTS path, rule to model input; read first |
| [Memorization literature](../notes/memorization_literature_2026_09_24.md) | NOTE (2026-09-24) | Ranked literature mapped onto MASSter; decision C |
| [Generalization methods](../notes/masster_generalization_methods_2026_09_27.md) | NOTE (2026-09-28) | Consistency, SAM, logit-regularization experiments with stop gates |
| [Displacement literature](../notes/intensity_displacement_literature_2026_09_24.md) | NOTE (2026-09-27) | Literature on Phase 0 results; things it argues against |
| [Excision literature](../notes/masster_excision_literature_2026_09_28.md) | NOTE (2026-09-28) | Internal-group excision and atom transfer |
| [SMARTS rule catalog](../reference/masster_smarts_rules.md) | REFERENCE (2026-10-01) | Registered, disabled and added rules |

## Key artifacts

- **Checkpoints (local `saved_ckpts/`):** `i1_inst_ob_obf_i50ig7rf/ckpt/` (best epoch 59, last 63), `i3_k1_ema_ob_obf_xypkkfti`, `fa1_full_attention_ob_obf_0vn8m2gr`. `i4` (`rd9rsdqx`) and `ce1`/`ce2` are not local. `FT`/`OB`-era checkpoints need `scripts/debug/masster_memorization/patch_inst_embedding.py` or `patch_ft_to_ob_ckpt.py` before use.
- **W&B (project `frag-gnn/frag-gnn`):** `i1` `i50ig7rf`; `i2` `jshvk4gv`; `i3` `xypkkfti`; `i4` `rd9rsdqx`; `fa1` `0vn8m2gr`, `fa1_s2` `2v2e1yod`; `mz1` `tck857p7`; `mznl1` `yp3wj9fa`; `nlpe1` `pqqfyvwo`; `cg1` `c8k6e244`; `sd1` `f3fu90zn`; `ce1` `u7ew4zpg`, `ce2` `u9auh1b9`; JEPA `j1a` `q047zga5`, `j1b` `xodee5pv`, `j2` `gzywzk39`.
- **Configs:** `config/masster/mh_ft_memorization_route_2026_09_24/` (all `*_ob_obf.yml` arms plus `fc1`, `fc1p`, `so1`); `config/masster/mh_smarts_pathway_2026_09_29/`; submitter `scripts/manual_slurm/fir/submit_masster_mh_ft_memorization_route.sh` (stages `rerun`, `graph`, `smarts`).
- **Scripts:** `scripts/debug/masster_memorization/` (base-peak, fragment-pair, post-cut, reference-gate, cut-probe, peak-ranking, dropout-view, local-context probes and their tests), `scripts/debug/smarts/`, `scripts/debug/isotopes/`.
- **Data:** `data/split/nist20_inchikey` (60/20/20 by 2D InChIKey); local `spec_df.pkl` carries `ORB`/`ORB_F` (NIST20 md5 `ce029870badc9e9e05935a4424f44c8d`).
- **Cited output dirs (git-ignored):** present locally: `output/ce_encoder_wandb_read_2026_10_01/`, `output/single_model_priority_read_2026_10_01/`, `output/i1_worst_predictions_2026_09_30/`; cited but absent on this checkout: `output/energetics_feasibility_2026_10_02/`, `output/energetics_joint_states_2026_10_02/`.
