# MASSter Ion-Mode Proposal, H Selection, and Intensity-Head Plan — log

> **Status:** REFERENCE · **Updated:** 2026-10-06
> Verbatim dated log moved from [masster_ion_h_channel_selection_plan.md](../plans/masster/masster_ion_h_channel_selection_plan.md) on 2026-10-06; the plan holds the current summary.


## CMF candidate-universe alignment (2026-09-14)

**Implemented; no training or oracle-coverage evaluation run.** User authorization
superseded the prior proposal-only restriction: MASSter now emits the full configured
H-transfer ladder for chemically valid, non-degenerate CMF adducts, matching FraGNNet.
It still removes CMF entirely for `[M+H]+` and `[M-H]-` when their base CMF ion duplicates
a reachable CRF H shift, and it retains the shared exclusion of in-source-loss precursors.
The pre-batch chemical-support filter uses the same mode/H validity contract.

This changes the MASSter candidate universe for carrier-switching adducts such as
`[M+Na]+` and `[M+Cl]-`. Existing checkpoints did not train the newly valid CMF-H channels,
so they are diagnostic-only; rebaseline candidate coverage and train matched controls before
scientific comparison. The adduct-stratified zero-only versus full-range oracle measurement
remains required to quantify the effect.

**Matched control prepared 2026-09-16, not yet launched.** The
[corrected unmerged CMF anchor](../plans/masster/masster_improvement_plan.md#corrected-unmerged-cmf-anchor-prepared-2026-09-16)
is the first config that trains these channels: four adducts on
`data/split/nist20_ma_mi_inchikey`, full acquisition context including dual CE, and channel
diagnostics on. Constructed from the resolved config, `[M+Na]+` and `[M+Cl]-` carry a live
nine-shift CMF ladder while `[M+H]+` and `[M-H]-` stay suppressed at duplicate CRF shifts `-1`
and `+1`; those two adducts are 38,156 and 7,820 of the split's 480,002 spectra. The training
loop logs no adduct-stratified metric, so the oracle comparison stays a post-hoc evaluation.

## H-support ablation measurement (2026-09-14)

**Read from the authenticated W&B API on 2026-09-14. No training, evaluation, or code ran for
this record.** The matched arms were launched 2026-09-08 and every one of them terminated at the
three-hour scheduler limit after 43 epochs (epochs 0-15 proposal-only), single seed `397623`,
merged NIST20 `[M+H]+` scope, U256/E1024, dense evaluation, hard thresholds zero.

| Arm | Config | Dense run | Subformula run |
| --- | --- | --- | --- |
| Gated | [warmup_loss_transform](../../config/masster/merged/warmup_loss_transform/nist20_merged_independent_jss_warmup_e0_15_dense.yml) | [`gbdszzlw`](https://wandb.ai/frag-gnn/frag-gnn/runs/gbdszzlw) | [`elfzwq6s`](https://wandb.ai/frag-gnn/frag-gnn/runs/elfzwq6s) |
| H supervision only | [support_ablation](../../config/masster/merged/support_ablation/nist20_merged_independent_jss_h_supervision_only_warmup_e0_15_dense.yml) | [`sr9xfkk8`](https://wandb.ai/frag-gnn/frag-gnn/runs/sr9xfkk8) | [`1fp53ss4`](https://wandb.ai/frag-gnn/frag-gnn/runs/1fp53ss4) |
| No H support | [support_ablation](../../config/masster/merged/support_ablation/nist20_merged_independent_jss_no_h_support_warmup_e0_15_dense.yml) | [`s23w0lt2`](https://wandb.ai/frag-gnn/frag-gnn/runs/s23w0lt2) | [`ynhectw6`](https://wandb.ai/frag-gnn/frag-gnn/runs/ynhectw6) |

### Channel selection is inert in every trained arm

Final-epoch validation summaries, dense arms:

| Validation metric | Gated | Supervision only | No H support |
| --- | ---: | ---: | ---: |
| `valid_ion_h_per_fragment_mean` | 8.5433 | 8.5438 | 8.5414 |
| `attributed_ion_h_per_fragment_mean` | 1.0198 | 1.0168 | 1.0169 |
| `retained_ion_h_count` / `valid_ion_h_count` | 1475.5 / 1475.5 | 1496.6 / 1496.6 | 1478.7 / 1478.7 |
| `retained_ion_h_probability` | 1.0000 | 1.0000 | 1.0000 |
| `ion_h_cutoff_pruned_fraction` | 0.0000 | 0.0000 | 0.0000 |

Retained equals valid in every arm. With `min_ion_h_relative_probability: 0.0`,
`min_ion_h_absolute_probability: 0.0`, and `masster_apply_channel_selection_in_training:
False`, what has been trained so far is an H-**support** head; the hard channel-selection stage of
the hierarchy has never been exercised in training or validation. About 8.5 channels are emitted
per fragment where roughly 1.02 are attributed. Phase 2's fixed-checkpoint sweep remains the first
measurement of the selection itself, not a refinement of an existing measurement.

### Spectral quality does not separate the arms

`val_cos_sim_0.01`, mean over the last five shared epochs (38-42), with the per-epoch standard
deviation over the last ten epochs as the noise scale:

| Arm | Dense | sd | Subformula | sd |
| --- | ---: | ---: | ---: | ---: |
| Gated | 0.7630 | 0.0047 | 0.7692 | 0.0049 |
| H supervision only | 0.7655 | 0.0055 | 0.7693 | 0.0043 |
| No H support | 0.7631 | 0.0054 | 0.7504 | 0.0132 |

Every dense difference is smaller than one per-epoch standard deviation of a single seed. The soft
intensity gate has no measured spectral benefit in either data variant, and no benefit in
validation `fragment_weighted_precision` either (0.6513 gated, 0.6517 supervision only, 0.6547 no
support). The one large gap, subformula without H support, comes from the least stable arm in the
set and is not yet a result. Best-epoch values (0.7668, 0.7706, 0.7694 dense) reorder the arms
relative to the five-epoch means, which is itself evidence that best-epoch selection sits inside
the noise band here. Paired seeds are required before the gate or the support BCE is credited or
removed on quality grounds.

### The support loss does sharpen channel ranking

Final-epoch validation, dense arms:

| Validation metric | Gated | Supervision only | No H support |
| --- | ---: | ---: | ---: |
| `required_ion_h_top_k_95_mean` | 3.2721 | 3.2842 | 3.6623 |
| `required_ion_h_top_k_95_p90` | 5.5665 | 5.5731 | 6.3422 |
| `required_ion_h_top_k_95_max` | 7.2532 | 7.2250 | 7.8719 |

Supervision concentrates 95% of per-fragment channel mass into about 11% fewer channels while
cosine and weighted precision stay flat. This is the separation the plan's goal section predicts:
a mechanism diagnostic improves while thresholded spectral quality does not. It supports carrying
the supervision into Phase 2 as a ranking signal, not as a spectral-quality intervention.

**Metric check required before these numbers gate anything.** `required_ion_h_top_k_95_mean` and
`required_ion_h_top_k_99_mean` agree to four decimals in all three arms (3.2721 versus 3.2722 in
the gated arm). Two different cumulative thresholds yielding the same mean needs verification in
the metric implementation before either is used as a Phase 4 gate.

### Clone initialization and self-gating (code check 2026-09-15)

The separate support projection starts with an exact copy of the intensity projection's
weight and bias. Loading an old checkpoint without support keys also copies its intensity
weights into the newly enabled support head. Both projections consume the same `tokens`:
see [head construction and checkpoint migration](../../src/fragnnet/model/bond_fragment_heads.py)
and the copy-initialization requirement in this plan's experiment matrix. This is deliberate
RNG-preserving initialization, not an independently initialized support signal.

At that point, `ion_h_support_logits == ion_h_logits == z`, so the independent intensity
path computes `softplus(c + z) * sigmoid(z)`, where `c` is the corrected fragment logit;
see [spectral weighting](../../src/fragnnet/model/masster_model.py). The initial gate is
therefore a self-gate. The parameters remain separate and can diverge through support BCE
and through the different spectral gradients entering the intensity and gate branches.
Clone initialization is a candidate confound for the measured gate null, not an established
explanation: the recorded runs do not measure projection divergence or logit correlation.
An initialization comparison would need matched RNG isolation and separate treatment of
legacy-checkpoint migration; no initialization or compatibility behavior changed here.

### Scope consequences

- `spec_params.prec_types` is `['[M+H]+']` only, and its CMF is removed as degenerate with
  reachable CRF `delta_h = -1`. The CMF candidate-universe alignment above therefore does not
  change this screen's candidate set, and cannot explain metric differences between 2026-09-08
  and 2026-09-14 runs on it.
- Arm C stays untestable in this scope, exactly as Phase 1 predicts. Ion-mode work requires
  `[M+Na]+` or `[M+Cl]-` in `spec_params.prec_types`.
- In this six-run ablation, `cmf_precursor_valid` is false for the only configured adduct.
  With `max_h_transfer = 4`, each fragment has at most nine valid CRF H channels (-4 through
  +4), with chemical validity reducing the measured mean to about 8.5. The `ion_h` metric
  names describe a general tensor interface; these results measure H ranking/support only,
  and provide no evidence of learned ion-mode discrimination. This scope statement does
  not establish the adduct scope or validity behavior of every historical trained run.

### Next concrete check

Script update (2026-09-15):
[`masster_ion_h_top_k_sweep.py`](../../scripts/eval/masster_ion_h_top_k_sweep.py)
now accepts `--support_absolute_thresholds` separately from final-intensity
`--joint_relative_thresholds`. It uses the model's existing one-channel-per-spectrum
fallback, requires a separate binary support head, includes selected-output weighted
precision/recall, and reseeds each rule. Use the newer checkpoint's matching training
YAML to preserve architecture and target construction. Disable fragment pruning with
`--fragment_relative_thresholds 0`; compare `--ion_h_top_ks 0 4 5` against a support-floor
sweep, with `--joint_relative_thresholds` supplied without values to omit intensity-floor
rules. Metrics are validation-equivalent and do not include a separate exported-intensity
cutoff pass. No checkpoint/data evaluation ran for this update.

1. Set `masster_gate_independent_intensity_with_support: False` in the H-support configs. No
   measured outcome pays for the gate, and it confounds every later support comparison.
2. Recover one trained checkpoint from the cluster and run the Phase 2 staged selection sweep on
   it. This is blocked, not deprioritized: W&B lists only `run-*-history` artifacts for these
   runs, so no checkpoint is recoverable from the run records, and the earlier `tq4bdn3g` sweep
   predates the chemistry corrections.
3. Keep the support BCE at `0.1` until Phase 2 measures whether its sharper ranking survives hard
   selection.
