# MASSter on NIST23 EI (nominal mass)

> **Status:** HUB · **Updated:** 2026-10-06
> MASSter on the NIST23 electron-ionization library (`[M]+`, 70 eV, unit resolution, natural isotope envelopes): plan, performance audit, screens.

## Where it stands

- **Best result:** e1 (natural isotopes), [W&B `jeljdikj`](https://wandb.ai/frag-gnn/frag-gnn/runs/jeljdikj) epoch 37, `config/masster/ei/e1_nominal_natural_isotopes.yml`: full `nist23_ei_inchikey` test (51,685 spectra) mean 1 Da cosine 0.869984, Stein weighted similarity 0.886845; 0.899167 on the 26,372 RASSP-eligible spectra (RASSP FormulaNet reports 0.929 on its own NIST17 set, descriptive only).
- **Isotope effect (one seed):** mean `val_cos_sim_1.0` over epochs 54-63 is 0.867738 (e1) vs 0.859883 (e0 `he3j5y3o`, no isotopes), +0.0079.
- **Where the gap is:** a 512-spectrum support oracle scores 0.99615 vs the model's 0.88703, so the gap is intensity allocation, not candidate support. Output power/mass tilts do not help. Stein falls from 0.9202 (2-16 atoms) to 0.8556 (>64).
- **Prepared, not launched:** matched `e2` (fresh control), `e3` (full attention), `e4` (EMA 0.999), W&B group `nist23_ei_intensity_2026_10_01`, same seed 397623, 64 epochs.
- **Uncommitted/unverified:** local isotope-offset and SHR-parsing audit fixes are not verified in the remote e0/e1 runs (no Git revision in remote metadata). No submitter stage for `config/masster/ei/`; the data may not be on Fir.
- **Next action:** record source provenance, check full-attention batch-512 memory, then run e2/e3/e4; add seeds before claiming the isotope effect. Then a matched RASSP comparison (same split, same Stein).

## Open threads

| Thread | Status | Next concrete step |
| --- | --- | --- |
| [Matched intensity arms e2/e3/e4](../plans/masster/masster_ei_nominal_plan.md#matched-intensity-configs-2026-10-01-prepared-not-launched) | PREPARED | Launch with provenance; compare cosine and Stein by size and on the RASSP subset |
| [Matched RASSP comparison](../findings/masster/masster_ei_performance_audit_2026_10_01.md) | OPEN | Check training overlap and preprocessing; run both models on one eligible split |
| Mass-aware intensity objective | PROPOSED | Test only if matched arms leave a Stein gap (audit priority 3) |
| H-shift width +/-6 (trained ablation) | OPEN | None until a trained-width ablation is wanted; keep +/-4 |
| NEIMS baseline, EI library metrics (composite cosine, retrieval) | NOT BUILT | Build NEIMS on `nist23_ei_inchikey` at 1 Da |
| Large-molecule errors | OPEN | Larger val sample with size/chemistry strata before changing candidates, H shift or SMARTS |

## Settled — read before re-proposing

- **Setup decisions (user):** nominal `nist23_ei`, not the SHR set; select on `val_cos_sim_1.0`; model isotopes; `loss_tolerance_abs` 0.5 ([plan](../plans/masster/masster_ei_nominal_plan.md)).
- [EI SMARTS screen](../findings/masster/masster_ei_smarts_screen_2026_10_01.md) — CLOSED NULL: anisoyl fluoride and para-hydroxyazobenzene add zero coverage; sulfonamide O-transfer +2.414% on seven train molecules, no val examples.
- [H-shift screen](../plans/masster/masster_ei_nominal_plan.md#ei-h-transfer-width-screen-2026-09-30-session-retain-4-for-now) — keep +/-4: L1 coverage 0.9040 at +/-2, 0.9339 at +/-3, 0.9482 at +/-4; +/-6 adds 1.31 points and 0.00314 cosine ceiling.
- [e1 annotation read](../plans/masster/masster_ei_nominal_plan.md#e1-checkpoint-annotation-read-2026-10-01-4-used-6-unresolved) — H+/-4 carries 3.10% of predicted intensity; dropping it costs 0.0010 cosine.
- [Fragment-count penalty](../plans/masster/masster_ei_nominal_plan.md#fragment-count-penalty-decision-2026-10-01-do-not-add-yet) — do not add: the loss cannot see which fragment explains a peak.
- [Test metrics by size](../plans/masster/masster_ei_nominal_plan.md#full-test-performance-by-molecule-size-2026-10-01) — unweighted cosine nearly flat by size; Stein declines. >48 atoms is extrapolation, not a RASSP-matched subset.
- [Output tilts](../findings/masster/masster_ei_performance_audit_2026_10_01.md) — peak powers 0.6-1.4 and m/z^alpha tilts all score below 0.88703.
- Related MASSter conventions live in [MASSter core](masster_core.md); EI SMARTS Enveda per-class gate is in [MASSter generalization](masster_generalization.md).

## All documents

| Doc | Status (updated) | One-line content |
| --- | --- | --- |
| [Plan: MASSter on NIST23 EI](../plans/masster/masster_ei_nominal_plan.md) | ACTIVE (2026-10-06) | Stage 0-1, W&B completion, audits, H-shift and SMARTS screens, size tables, e2-e4 configs |
| [EI performance audit](../findings/masster/masster_ei_performance_audit_2026_10_01.md) | FINDING (2026-10-06) | Stein vs RASSP, support oracle, replicate pairs, priorities |
| [EI SMARTS screen](../findings/masster/masster_ei_smarts_screen_2026_10_01.md) | CLOSED NULL (2026-10-06) | Three literature families, none admitted |
| [Isotope envelope plan](../plans/masster/masster_isotope_envelope_plan.md) | owned by generalization | Closed null for MS/MS; EI isotopes are natural (opposite case) |

## Key artifacts

- W&B: e1 `jeljdikj` (best ep 37), e0 `he3j5y3o` (best 0.861611 at ep 57).
- Configs: `config/masster/ei/{e0_nominal_no_isotopes,e1_nominal_natural_isotopes,e2_nominal_control,e3_nominal_full_attention,e4_nominal_ema999}.yml`.
- Data: `data/proc/nist23_ei/`, `data/split/nist23_ei_inchikey` (154,981 / 51,700 / 51,685), raw `data/df/nist23_ei_df.csv`.
- Outputs (ignored): `output/masster_ei_jeljdikj_analysis_2026_10_01/` (`test_all_stein/size_performance_stein_bins.csv`, `sample512_cosine/`), `output/masster_ei_smarts_2026_10_01/`, `output/masster_ei_nominal_2026_09_30/`.
- Scripts: `scripts/debug/ei/{debug_masster_ei_nominal_coverage,debug_masster_ei_smarts_screen,analyze_masster_ei_checkpoint,plot_masster_ei_size_performance}.py`; test `tests/test_debug_masster_ei_smarts_screen.py`.
