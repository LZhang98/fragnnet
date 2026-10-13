# MASSter MSGYM training arms

> **Status:** PREPARED · **Updated:** 2026-10-06
> Configs for the MSGYM k1/EMA/Fourier control and its arms (10-arm screen, sqrt, calibrated-likelihood,
> U256/E1024, 48-trial sweep) are built and resolve; none has been submitted. Created on 2026-10-06
> to hold facts that previously lived only in the project-state handoff log.

## Objective

Find a MASSter MSGYM recipe that raises linear `val_cos_sim_0.01` and MSGYM retrieval. The
generalization gap is the target: `ll04i00m` has train 0.884 against val 0.584 (63% of its
emitted-bin ceiling, see the [oracle ceiling](../../findings/masster/spectraverse_oracle_ceiling_2026_10_03.md)).
Reference points on the 9,954-spectrum simulation test set: `ll04i00m` 0.5713, GLACIER released
checkpoint 0.5455
([finding](../../findings/masster/masster_msgym_vs_glacier_released_ckpt_2026_09_24.md)).
Retrieval is read with the [one-command runner](../../guides/run_msgym_retrieval.md).

## Scope decision (user, 2026-09-24, from project-state handoff)

Benchmark subset only; no extra MassSpecGym data. `ms_gym_benchmark_1_5` is exactly the official
simulation-challenge set (`simulation_challenge == True`, 119,029 `[M+H]+` spectra, all with CE;
118,991 survive processing). The other 112,075 spectra lack CE on 97.6% of rows. The
`ms_gym_all_1_5` train fold would add 9,313 `[M+H]+` molecules (35,731 spectra, +76% over the
benchmark's 12,304; none in benchmark val/test by 2D InChIKey; only 21 have a CE, and a missing CE
becomes −1, a dedicated learned embedding under `abs-sines`). GLACIER trained on all 231,104 spectra
for retrieval and does not say which test population its 0.58 cosine covers.

## Baseline history (from project-state handoff)

### Config moved to the best measured recipe (2026-09-23)

`config/masster/massspecgym/msgym_benchmark_1_5_unmerged_dual_ce_fast_batch512_cuda.yml` had never
been trained. It now resolves to `c4a_base_s1` + `a1_prenorm` + the `d3` cosine schedule (warmup 1000,
floor 1%), MSGYM data, `[M+H]+` only, no CMF, `nce_max: 300`; 128 epochs of 194 steps
(`lr_total_steps: 24832`, was 174/290 constant LR). Unmerged is kept because the benchmark scores
per spectrum. A 2-epoch smoke on 256 spectra finished (9.2M parameters). Caveats: `a1`+`d3` together
untested, `d3` n=1. The unmerged NIST20 parent `6ks7v8ce` took about 5 h for that step count on an
L40S (0.73 s/step, 26.6 GiB peak). Generate on Fir with `setup_job_cc.py` (`fir:h100-3g.40gb`,
`j=<job_name>` to resume).

### GLACIER-parity features arm: RAN, no gain (W&B read 2026-09-24)

`ll04i00m` (base + `a_fused_ring`, `b_ring_size`, `b_fused_ring`, `c_nl_composition`; +11k
parameters; config `..._cuda_glacier_feats.yml`) against `1imawkx1` (the base):

- 118-127 val `cos_sim_0.01` tail 0.5669 against 0.5695 (−0.0026); below it at 101 of 128 epochs.
- Best checkpoints 0.5705 (epoch 60) against 0.5754 (epoch 62). Both are n=1, no MSGYM seed replicate.
  `1imawkx1` has no uploaded checkpoints; `ll04i00m` does.
- Both peak near epoch 60 and drift down: the recipe is past its best by mid-schedule.
- Overfit timing in `1imawkx1`: val proposal loss lowest at epoch 7, val primary loss
  (`coverage_weighted_spec_ce`) at epoch 14 (−1.793, ending −1.580), val cosine peaks at 62 while train
  loss falls to epoch 127. The CE rise is not coverage (`val_wrecall` flat 0.873-0.877 from epoch 8);
  `val_wprecision` climbs 0.59 to 0.71, consistent with sharpening (a temperature fit would test it).
  NIST `3bej6fq7` shows the same shape more mildly (lowest at 27, then +0.027).
- A `glacier_feats_fourier` config also exists; no read is recorded.
- EMA arm `..._cuda_ema999.yml` (base + `ema_decay: 0.999`, `upload_checkpoints` on) was built, not
  submitted; first EMA run with LR decay (both NIST wins used constant LR). Superseded by the k1 control below.

### How ms-pred baselines handle missing CE (re-read 2026-09-28, code at `318bb7b`)

- **ICEBERG 2.0 as published:** known-CE spectra only for training and testing (supplement §S2).
- **ICEBERG 2.1 (`main`):** `msg_simulation` keeps rows whose CE parses (119,029 expected, asserts
  none imputed). `msg_all` (retrieval) imputes by pseudo-labelling: a known-CE teacher predicts each
  missing-CE spectrum at NCE 5-150 step 5, the best entropy-similarity NCE wins, the student retrains
  on real plus pseudo labels. The imputation leaks the answer into retrieval: every missing-CE row is
  imputed from the prediction for its true SMILES, and stage 05 rewrites the test candidate tables so all
  256 candidates are scored at the CE that best fits the true molecule. Size unmeasured; no confidence gate.
- **GLACIER:** no imputation; NaN CE takes a fixed all-zero CE embedding (`collision_embed_merged`).
  The paper does not say how missing CE was handled.

## Arms

All configs are in `config/masster/massspecgym/` and are the `…_k1_ema999_fourier.yml` control plus
the listed keys. Each resolved diff was checked to be only those keys plus W&B labels.

### Control with the `i3` stack (2026-10-02)

`msgym_benchmark_1_5_unmerged_dual_ce_fast_batch512_cuda_k1_ema999_fourier.yml` = the `1imawkx1` base
(already pre-norm, so the model is `i4` on MSGYM) + `k1`'s two 0.2 dropouts, `ema_decay: 0.999`,
`int_embedder: fourier`, checkpoint upload; keeps the 128-epoch cosine schedule. Local 2-epoch smoke
on 512 spectra per split exits 0 (EMA state and a `Linear(22, 32)` Fourier table on both CE channels).
No MSGYM seed spread exists. Read: best checkpoint on test against `ll04i00m` 0.5713 and GLACIER 0.5455.

### 10-arm screen (2026-10-03), prepared, not submitted

Suffix on the control; goal is the generalization gap.

| Arm | Change | Params |
|---|---|---|
| `s2` | seed replicate (first MSGYM seed spread) | 9.23M |
| `wd10` | weight decay 0.1 | 9.23M |
| `e64` | 64-epoch cosine | 9.23M |
| `lr5e4` | LR 5e-4 | 9.23M |
| `b256` | batch 256, steps doubled | 9.23M |
| `enc256` / `enc512` | `mol_hidden_size` (also sizes bond, boundary and charge heads: whole-model width) | 5.76M / 13.89M |
| `mol3` | encoder depth 3 | 10.12M |
| `bond1` | bond head 3 to 1 layers | 7.16M |
| `itf1` | intensity transformer 2 to 1 layers | 8.18M |

Checks: every config builds on CPU; no training smoke. Next: submit with the control and read each arm
against |control − `s2`|.

### Sqrt-intensity arm (2026-10-03)

`…_k1_ema999_fourier_sqrt.yml` = control + `ints_transform: sqrt` + `masster_merge_same_mz_rows: true`.

- **Why.** GLACIER's retrieval edge on entropy similarity sits in predicted peaks below 1% of base,
  which sqrt-intensity training weights and MASSter's linear density loss does not.
- **New flag (default off).** `masster_merge_same_mz_rows` merges same-m/z pathway rows at exact
  mass so evaluation squares each m/z's summed probability, not each pathway. `MassterPL` refuses a
  transformed `ints_transform` at exact mass without it. Tests `tests/test_masster_merge_same_mz_rows.py`;
  250 related tests pass.
- **Smoke.** Local 2 epochs on 512 spectra per split (batch 32 for the 8 GB card) exits 0; the
  checkpoint carries both keys; the export has no duplicate m/z rows.
- **Next.** Submit control and arm together; read linear `val_cos_sim_0.01`, then test retrieval with
  `scripts/eval/score_retrieval_candidates.py` against control and GLACIER.

### Calibrated-likelihood training arms (2026-10-03)

`…_k1_ema999_fourier_{pow075,pm1,sqrt_pm1,pow075_pm1}.yml` put the calibrated retrieval rule
(`ll_g0.75_t2_e1e-4_a0.5`, see the [retrieval finding](../../findings/masster/masster_msgym_formula_retrieval_full_2026_10_03.md#calibrated-likelihood-val-selected-one-test-read-added-2026-10-03))
into training.

- **Keys.** `ints_transform: pow0.75` is new (target y^0.75, untransform x^(4/3)).
  `output_distribution: peak_marginal` with `pm_tolerance_multiple: 1.0` replaces the Gaussian density
  by the predicted mass within 1x tolerance. Transformed arms set `masster_merge_same_mz_rows: true`.
- **Not trained.** The prediction exponent γ (cross-entropy is minimized at q = target weights, so
  flattening goes on the target) and the 1e-4 floor.
- **Smoke.** One CPU epoch on 64 spectra exits 0 for `pow075`, `pm1`, `pow075_pm1` (GPU was busy).
- **Read.** Linear `val_cos_sim_0.01`, then retrieval with the rule re-selected on val per checkpoint:
  a transformed arm's q is p^0.5 or p^0.75 of the exported p, so the frozen rule does not carry over.

### U256/E1024 follow-up and the 48-trial sweep

- Five batch512 128-epoch configs (`msgym_u256_e1024_{control,s2,wd10,sqrt,pow075_pm1}.yml`) are
  described in [the marginal-selection plan](masster_fragment_marginal_selection_plan.md#msgym-u256e1024-follow-up-2026-10-05-prepared-not-run).
  Next: GPU smoke and the matched U256/E4096 comparison.
- The [48-trial validation-loss sweep](msgym_val_loss_sweep_48_2026_10_03.md) (W&B `tiwhun9q`) holds
  linear Gaussian CE and U256/E4096 fixed and searches batch, depth, dropout, LR and weight decay.

## Open

Nothing above has trained. Per the handoff: submit the control and `s2` first (gives the
seed spread), then read each arm against it. Arms need the Fir login
and checkout; no cluster job has been submitted for any MSGYM arm.
