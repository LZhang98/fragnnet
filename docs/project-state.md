# Current project state

> **Status:** REFERENCE · **Updated:** 2026-10-06
> Dashboard: one block per workstream, replaced (not appended) on every handoff. Details live in
> the [workstream hubs](workstreams/); the pre-2026-10-06 dated handoff is frozen in
> [history/handoff_log.md](history/handoff_log.md).

Keep this file under about 200 lines. To hand off, update the hub's "Where it stands" and
"Open threads", then replace that workstream's block below
([documentation rules](rules/documentation.md#where-each-kind-of-fact-goes)).

## Workstreams

| Workstream | Hub | State |
|---|---|---|
| MASSter generalization on NIST20 (`i1` line) | [masster_generalization](workstreams/masster_generalization.md) | ACTIVE: computed-energetics gate is next |
| MASSter core (architecture, training, speed) | [masster_core](workstreams/masster_core.md) | Prepared arms; no scheduled work |
| MassSpecGym training and retrieval | [msgym](workstreams/msgym.md) | Many arms PREPARED, none submitted |
| Spectraverse v1.0.2 | [spectraverse](workstreams/spectraverse.md) | Five arms PREPARED, none submitted |
| NIST23 EI (nominal) | [nist23_ei](workstreams/nist23_ei.md) | e2/e3/e4 PREPARED |
| CASMI 2026, Enveda-180, shared data quality | [casmi_enveda](workstreams/casmi_enveda.md) | ACTIVE: CASMI train filter decision |
| FraGNNet and the DAG | [fragnnet](workstreams/fragnnet.md) | Dormant; Stage A PREPARED |

## MASSter generalization on NIST20

- **Best recipe:** `i4` = `i1` + `k1` dropout + EMA 0.999 + pre-norm, W&B `rd9rsdqx`, tail
  `val_cos_sim_0.01` 0.7729 (epochs 54-63): +0.0122 over `i1` (0.7607). Seed spread 0.0042.
- **Open problem is the floor:** 330 spectra / 119 molecules below 0.05 at instrument tolerance
  (train sample: 4); no tested arm moves it.
- **Prepared, not submitted:** `so1` (templates-on sulfonamide O transfer), `fc1`/`fc1p`
  (local screen not positive).
- **Next:** computed-energetics information gate. First fix a bounded state/calculation policy
  and a train/calibration/confirmation protocol.

## MASSter core

- **Screen baseline:** merged MA/MI 128-epoch screen, control `c4a_base_s1` 0.7270 (spread
  0.0007, gate ±0.005). Real levers: `a1_prenorm` +0.0056, cosine LR decay `d3` 0.7357 (n=1).
- **Prepared:** full-length `wd10` and its `trainaux` control (`fir:h100-80gb`); `a4`,
  `a5_ce_pair_bias`, `b4`. Ion-H Phase 2 is blocked on a recoverable checkpoint.
- **Next:** an `a1_prenorm` seed replicate. Otherwise, core levers go through the `i1` stack.

## MassSpecGym

- **Best:** `ll04i00m` test `cos_sim_0.01` 0.5713 vs GLACIER released 0.5455. Formula-track
  retrieval hit@1 on test: 0.358 cosine, 0.485 log-likelihood, 0.528 calibrated rule.
- **Problem:** generalization, train 0.884 vs val 0.584; no MSGYM seed replicate exists.
- **Prepared, not submitted:** k1/EMA/Fourier control + 10-arm screen, sqrt arm, 4
  calibrated-likelihood arms, 5 U256/E1024 configs, 48-trial sweep `tiwhun9q`
  ([arms plan](plans/masster/masster_msgym_training_arms_plan.md)).
- **Next:** submit the control and `s2` on Fir; re-select `loglik_cal` on val, then run test
  retrieval for the best checkpoint.

## Spectraverse

- **Stuck at** `cos_sim_0.01` 0.458 (cv0 `ggifzj9m`, cv1 `u46v0k0g`). The causes are intensity
  allocation (62% of the emitted-bin ceiling) and 14% unexplainable spectra (MS-DIAL is 46% of
  those).
- **Prepared, not submitted (cv0/cv1):** `masster_c0rerun`, `short18`, `subceil05`,
  `merged_nomsdial`, `dedup`, plus the ten-fold `abs005`.
- **Next:** copy `data/split/spectraverse_benchmark_{subceil05,nomsdial,dedup}` to Fir, launch,
  read against `masster_c0rerun`.

## NIST23 EI

- **Best:** e1 `jeljdikj`@37, test 1 Da cosine 0.869984, Stein 0.886845. Isotopes are +0.0079
  over e0 (one seed). The gap is intensity allocation (support oracle 0.996).
- **Next:** record provenance, check full-attention batch-512 memory, run e2/e3/e4, then a
  matched RASSP comparison.

## CASMI 2026, Enveda-180, shared data

- **CASMI 2026:** extracted, not preprocessed. The visible test is a copy of train, so validate
  molecule-disjoint. Dedup drops 17.6% of train; NIST HR would add 34,295 molecules.
- **Enveda-180:** baseline (standalone, ACE only, no EMA) prepared, not submitted. The combined
  PCDL+NIST23+Enveda-180 split is a manifest only.
- **Next:** decide the CASMI train filter, then preprocess.

## FraGNNet

- **Dormant since September.** Best `pqx7ywc9` 0.7555 `cos_hun_inst_tol` (NIST20 test). The DAG
  oracle is 0.937, so the gap is intensity ranking, not coverage.
- **Prepared:** FT contender-pair Stage A (`c0`/`c1`/`p1`/`p2`). Re-measure steps per epoch
  after the 2026-09-26 sampler fix before submitting.

## Uncommitted working tree (2026-10-06)

- Retired the empty `docs/spectra_cleanup.md` placeholder; the documentation index now links
  directly to the implementation and canonical finding. Local link checks and
  `git diff --check` passed.

## Standing cross-workstream rules

- Merged and per-CE evaluation setups are not comparable; neither are runs across the
  2026-09-03 target change, the `FT` → `ORB`/`ORB_F` relabel, or different binning (0.01 vs 0.1 Da).
- Read a hub's "Settled" list before re-proposing a mechanism: most closed nulls are recorded there.
- Cluster procedure: [submit_cluster_runs guide](guides/submit_cluster_runs.md). Artifact locations:
  [artifact rules](rules/artifacts.md).
