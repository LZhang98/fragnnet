# MASSter generalization on NIST20 (the `i1` line)

> **Status:** HUB · **Updated:** 2026-10-18
> Why MASSter fits training molecules and ranks new ones poorly on unmerged NIST20 `[M+H]+`: the memorization routes, `i1` error analyses, information gates and the SMARTS/excision/isotope/JEPA side threads.

## Where it stands

- **Prepared, not submitted:** [rare-element route 1](../plans/masster/rare_chemistry_augmentation_plan.md#rare-element-handling-revised-2026-10-14): `a_periodic` atom feature and a hidden-Br test, A0/E0/E1/E2/E3 on `i1` (E3 = E1 without `a_onehot`, added 2026-10-15) (`config/masster/rare_chemistry_2026_10_10/`, launcher `scripts/manual_slurm/fir/submit_masster_rare_chemistry.sh`). The A1/A3 sampling arms were removed 2026-10-14 without running ([class census](../findings/masster/rare_chemistry_censuses.md#rare-chemistry-census-a1-moves-exposure-a3-does-not), [element-swap census](../findings/masster/rare_chemistry_censuses.md#rare-chemistry-element-swap-census-halogens-have-measured-pairs-seassi-do-not)).
- **Best recipe:** `i4` = `i1` (instrument input on `ORB`/`ORB_F`) + `k1` dropout 0.2 + weight EMA 0.999 + intensity pre-norm. Tail mean of `val_cos_sim_0.01_epoch/mean`, epochs 54-63: **0.7729** (W&B `rd9rsdqx`), +0.0019 over `i3` (`xypkkfti`, 0.7710) and +0.0102 / +0.0122 over `i1` (`i50ig7rf`, 0.7607). Seed spread 0.0042; single-seed gate +0.005. Configs: `config/masster/mh_ft_memorization_route_2026_09_24/{i1,i3,i4}_*_ob_obf.yml`.
- **Open problem:** the validation floor, not the average. 330 spectra / 119 molecules score below 0.05 at instrument tolerance (train sample: 4); 88% are ranking failures with the candidates present. No tested arm (`fa1`, `i3`, cut probe, peak-ranking loss, fragment connectivity) moves a floor chosen by an independent checkpoint.
- **Single next action:** the computed-energetics information gate. Formal floor coverage is 97% but only 14 / 24 harder channels keep a distance-preserved product pair, so first fix a bounded, target-independent state/calculation policy and a train/calibration/confirmation protocol ([plan](../plans/masster/masster_memorization_route_plan.md#computed-product-ion-energetics-information-gate-2026-10-01-proposed)).
- **Prepared, not submitted:** `md05` = `i4` + `mol_dropout` 0.5 (GNN encoder dropout; user decision 2026-10-18), control `i4` `rd9rsdqx`, Fir stage `md05`, config `config/masster/mh_ft_memorization_route_2026_09_24/md05_mol_dropout05_ob_obf.yml`. Motivation: in the MSGYM sweep `tiwhun9q`, mean best val loss improved monotonically over the sampled `mol_dropout` 0.0–0.3, with the best at the grid edge ([sweep read](../plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#what-the-random-search-suggests)). No MASSter run on W&B has used more than 0.3 (152 runs checked), so 0.5 is untested. Next step if it passes: a joint NIST20 `i4` random search (LR, schedule, wd, dropouts, batch).
- **Prepared, not submitted:** `so1` (`i1` + SMARTS templates, aryl-sulfonamide O transfer; Fir stage `smarts`), `fc1`/`fc1p` (configs exist, not in the submitter; local screen not positive).
- **Branch `rare-chemistry-sampling`:** route 1 code, configs, launcher and censuses are committed there; JEPA code and data removal is committed (`6f2fdea2`).
- Core architecture/training lives in [masster_core](masster_core.md); retrieval in [msgym](msgym.md); other datasets in [spectraverse](spectraverse.md), [nist23_ei](nist23_ei.md), [casmi_enveda](casmi_enveda.md); the sibling model in [fragnnet](fragnnet.md).

## Open threads

| Thread | Status | Next concrete step |
| --- | --- | --- |
| [Rare chemistry augmentation](../plans/masster/rare_chemistry_augmentation_plan.md) | PREPARED | Rare-element route 1 (A0/E0/E1/E2/E3: `a_periodic`, hidden-Br test) ready to submit; sampling arms removed 2026-10-14 |
| [Memorization-route plan](../plans/masster/masster_memorization_route_plan.md) (dated log: [history](../history/masster_memorization_route_plan_log.md)) | ACTIVE | Energetics gate protocol (above); coverage run `so1` when the user decides |
| [Energetics feasibility](../findings/masster/masster_energetics_feasibility_2026_10_02.md) | FINDING | State/calculation policy, missing-feature handling, independent selector, effect size, before any fit |
| [Transfer-oriented (MLDG-style) training](../plans/masster/masster_memorization_route_plan.md#transfer-oriented-training-for-covered-ranking-failures-2026-10-01-deferred) | DEFERRED | Only a cheap head-gradient conflict check on training-only batches, if revived |
| [Intensity reliability](../plans/masster/masster_intensity_reliability_plan.md) | DEFERRED | P2-P5 not run; needs replicate provenance upstream |
| [Experiment reopening audit](../findings/experiment_reopening_audit_2026_09_28.md) | FINDING | Untested alternatives: out-of-fold ensemble distillation, narrower R-Drop and smooth-CE follow-ups |
| [Dropout-view disagreement](../findings/masster/masster_dropout_view_disagreement_2026_09_27.md) | FINDING | User choice: R-Drop arm pair on `i1`, or 8-view head averaging at inference (+0.0031, no training) |

## Settled — read before re-proposing

**User decisions**

- [`f0`/`f1` retired](../findings/masster/masster_memorization_audits.md#audit-of-every-attempt-to-close-massters-memorization-gap) — the 11 SUPERSEDED findings on old `FT` labels were removed 2026-10-18 (last version at `731fe493`, e.g. `git show 731fe493:docs/findings/masster/masster_ranking_decomposition_2026_09_23.md`); cite the `i1` reads, never `f0`/`f1`/`c0` numbers as current.
- [Worst predictions](../findings/masster/masster_i1_worst_predictions_2026_09_30.md) — rank by `cos_hun_inst_tol` (instrument tolerance), not 0.01 Da bins; 497 of 903 "worst" logged spectra score >= 0.5.
- [Memorization routes](../plans/masster/masster_memorization_route_plan.md) — try many routes, never a single-input block (identity reaches every head input); seed spread 0.0042; `i1` is the control for new arms.
- [SMARTS goal](../findings/masster/masster_smarts_findings.md#masster-smarts-rules-and-outlier-spectra) — raise the lower bound (outlier spectra); NIST20 only, no MONA; 8 rules commented out, re-check on Enveda-180.
- JEPA removal (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_jepa_embedding_plan.md`) — code, data and `dreams` env removed 2026-10-04 (user); other null features deliberately kept.
- Cross-molecule code (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_cross_molecule_supervision_gate.md`) — `cm0`-`cm2` reverted at the user's request; census and artifacts kept; never run.

**The train/val gap and what moved it**

- [`i1` memorization read](../findings/masster/masster_i1_memorization_read_2026_09_28.md) — instrument input narrows the eval gap 0.183 to 0.158; about 86% remains; Lumos bias fixed.
- [Attempts audit](../findings/masster/masster_memorization_audits.md#audit-of-every-attempt-to-close-massters-memorization-gap) — read first for any memorization arm; +0.003 gate is about one standard error.
- Plan review (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_memorization_plan_review_2026_09_27.md`) — separate held-out promotion from mechanistic criteria; route corruption measures dependence only.
- [Route plan, current summary](../plans/masster/masster_memorization_route_plan.md#current-summary) — `i3` +0.0102 and `i4` pass; full attention +0.0044 just under the gate; route removal, noise, masks, windows and m/z/loss positions all null or worse.
- [Literature](../notes/memorization_literature_2026_09_24.md) and [methods note](../notes/masster_generalization_methods_2026_09_27.md) — ranked list; decision C constant LR; R-Drop/SAM gates.
- [Dropout-view disagreement](../findings/masster/masster_dropout_view_disagreement_2026_09_27.md) — 8-view mean beats best flattened eval by +0.0031 on `i1`; centered-logit penalty not buildable as specified.

**`i1` error analyses (no missing input, hedging, transfer failure)**

- [Base-peak table](../findings/masster/masster_i1_base_peak_errors.md#wrong-base-peak-choices-of-i1-a-broad-transfer-deficit-not-one-missing-distinction) — base peak missed in 40% of val (train 15%); pair biases are hedging; no single missing distinction.
- [Residual structure](../findings/masster/masster_i1_residual_structure_2026_09_28.md) — no recorded metadata left; per-molecule energy scale +0.010 val but unpredictable; read before any new conditioning input.
- [Post-cut gate](../findings/masster/masster_i1_base_peak_errors.md#post-cut-information-gate-on-i1-explicit-fragment-and-charge-descriptors-add-nothing) — connectivity/charge descriptors add <= +0.001 over the `P^gamma` baseline.
- [`fc1` screen](../findings/masster/masster_i1_base_peak_errors.md#fragment-connectivity-branch-fc1-local-screen-against-i1-is-not-positive) — +0.0050 mean, below the 0.0076 control spread; no Fir run.
- [Fragment pairs](../findings/masster/masster_i1_fragment_pairs.md#ambiguity-preserving-base-peak-comparisons-for-i1) and [pair contrasts](../findings/masster/masster_i1_fragment_pairs.md#conditional-success-and-failure-contrasts-for-i1s-candidate-choices) — 73% of ranking pairs have several compatible atom sets; only familiarity replicates, no cut rule.
- [Isomer split Step 0](../findings/masster/masster_isomer_split_step0_2026_10_01.md) — `i1`/`fa1` agree on 85.9% of confident splits; cut-class prior adds nothing; no arm built.
- [Metadata usage audit](../findings/masster/masster_metadata_usage_audit_2026_10_03.md) — adduct and MSGYM fragmentation-mode embeddings are constants; compared with MassFormer/ICEBERG/FIORA/GLACIER.

**Closed information gates and probes (NIST20, frozen `i1`)**

- Cut-score probe (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_cut_intensity_probe_2026_10_01.md`) — additive boundary-score correction selects zero; closes this form, not every cut-supervision design.
- Peak-ranking pilot (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_peak_ranking_pilot_2026_10_02.md`) — tuning selects zero ranking weight; full-data end-to-end untested.
- Structural alignment (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_structural_alignment_audit_2026_09_29.md`) — exact connectivity keeps 0.85% of measured intensity on unfamiliar queries; stop.
- Reference information gate (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_reference_information_gate_2026_09_29.md`) — ion corrections select zero; loss alignment lowers cosine (0.6287 vs 0.6385).
- [Candidate correspondence](../findings/masster/masster_cross_molecule_supervision.md#deterministic-candidate-correspondence-coverage-before-a-competition-loss) — coverage advantage does not predict accuracy gain; transfer plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_candidate_transfer_plan.md`) STOPPED.
- [Cross-molecule gate](../findings/masster/masster_cross_molecule_supervision.md#cross-molecule-gate-transferable-spectral-relationships-weak-winner-labels) — soft relationship signal +0.050 at Tanimoto <= 0.4; hard shared-winner labels unsupported; never trained.
- Local context allocation (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_local_context_allocation_plan.md`) — frozen-token probe stopped at A1 epoch 17; local term <= +0.001.

**QET / NCE input and instruments**

- [Breakdown-curve form fit](../findings/breakdown_curve_form_fit_2026_09_26.md) — Arrhenius form fits the CE series (0.995); no survival head.
- [QET head gates and NCE input](../findings/masster/masster_qet_projection_candidate_union_2026_09_26.md) — smooth CE inputs fail; `n4` fourier +0.0053 over `c0`.
- [Orbitrap CE scales](../findings/nist_orbitrap_instrument_ce_scales_2026_09_28.md) — two ladders are Elite vs Lumos; `ORB`/`ORB_F` from raw MSP; `FT` and `OB`/`OB_F` refused. `ce1`/`ce2` (NCE into the encoder) +0.0003/+0.0005, no pass.

**SMARTS, excision, isotopes**

- [Rearrangement mining](../findings/masster/masster_rearrangement_mining_2026_10_14.md) / [rewire rules plan](../plans/masster/masster_rewire_rules_training_plan.md) — about 1% of [M+H]+ Orbitrap intensity needs a rearrangement, concentrated in a 1.1-1.9% tail of spectra; 13 reviewed mechanism rules with continuation reach 39% of it (null 7%) and lift the val-tail cosine ceiling from 0.652 to 0.782. PREPARED: built and tested; G0/G0b pass; arms a0/a0d1/a0null ready to submit (stage `rewire`).
- SMARTS pathway (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_smarts_pathway_plan.md`) / [gates](../findings/masster/masster_smarts_findings.md#smarts-pathway-gates-g0-g1-g2-g2b-every-arm-fails-its-gate) — CLOSED NULL: G0 +0.00125 < 0.003, G1 STOP, Track B 0 promoted; `p1` code stays built, off.
- [SMARTS CUDA audit](../findings/masster/masster_smarts_findings.md#masster-smarts-templates-cuda-enablement-audit-and-proposal-reach) — templates OFF by default; match standardized molecules; SMARTS vs CUDA-template mismatch is a hard error.
- [SMARTS on Enveda-180](../findings/masster/masster_smarts_findings.md#masster-smarts-rules-on-enveda-180-per-class-gate) — only S=O rules pass, 2-10x weaker than NIST20; none of the 8 disabled rules finds support.
- Excision (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_excision_continuation_gate_2026_09_28.md`) — only the aromatic nitro NO-loss family confirms (+0.648% class intensity); no channel built.
- Isotope envelope (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_isotope_envelope_plan.md`) — CLOSED NULL: perfect monoisotopic + best scalar envelope recovers 0.0047 cosine against a 0.005 gate.
- JEPA embedding (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_jepa_embedding_plan.md`) — CLOSED NULL: `j1a` +0.0000, `j1b` -0.0003, `j2` -0.0031 against `i1`.
- **Retired 2026-10-18:** 17 documents of this workstream removed; summaries and open items in [retired documents](../history/retired_docs.md#masster_generalization).
- Intensity displacement — retired 2026-10-18 while DEFERRED (`git show 95341fce:docs/plans/masster/masster_intensity_displacement_plan.md`); if revived: re-derive premises on `i1` first; one cheap hashed-environment retry of the bond-local learner is open.
- Ranking transfer — retired 2026-10-18 while DEFERRED (`git show 95341fce:docs/plans/masster/masster_ranking_transfer_plan.md`); if revived: EMA already carried into `i3`/`i4`; gates 2-3 unread, no step pursued.
- SMARTS pathway: the only open route is user decision D2, targeting the template subset (G0 subset value 0.022) instead of the whole split.
- Judge memorization arms on replicated held-out performance; gap closure and route sensitivity are separate mechanistic claims.

## All documents

| Doc | Status (updated) | One-line content |
| --- | --- | --- |
| [Rare chemistry augmentation](../plans/masster/rare_chemistry_augmentation_plan.md) | PREPARED (2026-10-14) | Shared NIST20/MSGYM rarity census, capped sampling, conditional replicate augmentation and confirmation gates; sampler removed 2026-10-14, route 1 prepared |
| [Memorization-route plan](../plans/masster/masster_memorization_route_plan.md) | ACTIVE (2026-10-06) | Route-removal arm queue, gates, current summary; open energetics and MLDG designs |
| [Memorization-route log](../history/masster_memorization_route_plan_log.md) | REFERENCE (2026-10-06) | Verbatim dated log of every arm, read and build (batches 1-5, `i1` stack, ladders, `ce1`/`ce2`) |
| [Intensity reliability plan](../plans/masster/masster_intensity_reliability_plan.md) | DEFERRED (2026-10-06) | Ladder-aware replicate noise, P1 census, uncertainty ensembles |
| [Rewire rules in training](../plans/masster/masster_rewire_rules_training_plan.md) | PREPARED (2026-10-15) | Mechanism rules with continuation as an after-selection training stage; G0/G0b pass; a0/a0d1/a0null configured, not submitted |
| [Rewire rules implementation spec](../plans/masster/masster_rewire_rules_implementation.md) | PREPARED (2026-10-15) | File-by-file spec of the built stage, its tests and deviations |
| [Cross-molecule supervision](../findings/masster/masster_cross_molecule_supervision.md) | FINDING (2026-10-18, merged) | Two parts: training-only census of transferable spectral relationships; endpoint and NCE-band candidate-correspondence coverage |
| [Rare chemistry censuses](../findings/masster/rare_chemistry_censuses.md) | FINDING (2026-10-18, merged) | Two parts: class-rarity census (A1 exposure shift, A3 fails support gate); element-swap pairs (57 Br→Cl training pairs on NIST20; none for Se/As/Si or on MSGYM) |
| [Dropout-view disagreement](../findings/masster/masster_dropout_view_disagreement_2026_09_27.md) | FINDING (2026-09-28) | R-Drop and logit-penalty pricing; 8-view averaging |
| [Energetics feasibility](../findings/masster/masster_energetics_feasibility_2026_10_02.md) | FINDING (2026-10-02) | xTB feasibility, joint-state coverage 97%, geometry limits |
| [`i1` base-peak errors](../findings/masster/masster_i1_base_peak_errors.md) | FINDING (2026-10-18, merged) | Three-step plan: base-peak table and hedging biases; post-cut/charge descriptors add nothing; `fc1` local screen not positive |
| [`i1` fragment pairs](../findings/masster/masster_i1_fragment_pairs.md) | FINDING (2026-10-18, merged) | Two steps: ambiguity-preserving pair table on 52,020 val spectra; matched success/failure contrasts |
| [`i1` memorization read](../findings/masster/masster_i1_memorization_read_2026_09_28.md) | FINDING (2026-09-28) | Route-map probe on `i1`; gap 0.158 |
| [`i1` residual structure](../findings/masster/masster_i1_residual_structure_2026_09_28.md) | FINDING (2026-09-28) | Where the val deficit is; no missing metadata |
| [`i1` worst predictions](../findings/masster/masster_i1_worst_predictions_2026_09_30.md) | FINDING (2026-10-06) | Floor of 330 spectra; ranking failure; `so1` arm |
| [Isomer split Step 0](../findings/masster/masster_isomer_split_step0_2026_10_01.md) | FINDING (2026-10-01) | Within-peak isomer split gate |
| [Memorization audits](../findings/masster/masster_memorization_audits.md) | FINDING (2026-10-18, merged) | Two parts: W&B read of the growing unmerged gap (2026-09-16); every arm aimed at the gap, re-derived from code and W&B (2026-09-27) |
| [Metadata usage audit](../findings/masster/masster_metadata_usage_audit_2026_10_03.md) | FINDING (2026-10-03) | CE/adduct/instrument/mode vs published models |
| [QET projection / candidate union](../findings/masster/masster_qet_projection_candidate_union_2026_09_26.md) | FINDING (2026-09-28) | QET head gates; NCE input; `n4` fourier |
| [SMARTS findings](../findings/masster/masster_smarts_findings.md) | FINDING (2026-10-18, merged) | Five parts: rule set and literature review; templates on CUDA and proposal reach; rules matter on 37 outlier val spectra; Enveda-180 per-class gate on all 41 rules; pathway gates G0/G1/G2/G2b all fail |
| [Rearrangement mining](../findings/masster/masster_rearrangement_mining_2026_10_14.md) | FINDING (2026-10-15) | Non-bond-cut peaks in MSGYM+NIST20 [M+H]+; tail profile; 13 gated mechanism rules |
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

- Results ledger: [results/nist20.md](../results/nist20.md), every measured result as one row linking its source.
- **Checkpoints (local `saved_ckpts/`):** `i1_inst_ob_obf_i50ig7rf/ckpt/` (best epoch 59, last 63), `i3_k1_ema_ob_obf_xypkkfti`, `fa1_full_attention_ob_obf_0vn8m2gr`. `i4` (`rd9rsdqx`) and `ce1`/`ce2` are not local. `FT`/`OB`-era checkpoints need `scripts/debug/masster_memorization/patch_inst_embedding.py` or `patch_ft_to_ob_ckpt.py` before use.
- **W&B (project `frag-gnn/frag-gnn`):** `i1` `i50ig7rf`; `i2` `jshvk4gv`; `i3` `xypkkfti`; `i4` `rd9rsdqx`; `fa1` `0vn8m2gr`, `fa1_s2` `2v2e1yod`; `mz1` `tck857p7`; `mznl1` `yp3wj9fa`; `nlpe1` `pqqfyvwo`; `cg1` `c8k6e244`; `sd1` `f3fu90zn`; `ce1` `u7ew4zpg`, `ce2` `u9auh1b9`; JEPA `j1a` `q047zga5`, `j1b` `xodee5pv`, `j2` `gzywzk39`.
- **Configs:** `config/masster/mh_ft_memorization_route_2026_09_24/` (all `*_ob_obf.yml` arms plus `fc1`, `fc1p`, `so1`); `config/masster/mh_smarts_pathway_2026_09_29/`; submitter `scripts/manual_slurm/fir/submit_masster_mh_ft_memorization_route.sh` (stages `rerun`, `graph`, `smarts`).
- **Scripts:** `scripts/debug/masster_memorization/` (base-peak, fragment-pair, post-cut, reference-gate, cut-probe, peak-ranking, dropout-view, local-context probes and their tests), `scripts/debug/smarts/`, `scripts/debug/isotopes/`.
- **Data:** `data/split/nist20_inchikey` (60/20/20 by 2D InChIKey); local `spec_df.pkl` carries `ORB`/`ORB_F` (NIST20 md5 `ce029870badc9e9e05935a4424f44c8d`).
- **Cited output dirs (git-ignored):** present locally: `output/ce_encoder_wandb_read_2026_10_01/`, `output/single_model_priority_read_2026_10_01/`, `output/i1_worst_predictions_2026_09_30/`; cited but absent on this checkout: `output/energetics_feasibility_2026_10_02/`, `output/energetics_joint_states_2026_10_02/`.
