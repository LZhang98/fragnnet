# Memorization read on `i1`: the instrument input narrows the train/val gap

> **Status:** FINDING · **Updated:** 2026-09-28
> Inference only, on the local RTX 2070 SUPER. The checkpoints are `i1_inst_ob_obf` (`i50ig7rf`,
> `n4` on the relabelled `OB` / `OB_F` NIST20) at its best epoch (59) and its last (63). The probe
> is the one `c0` ran on 2026-09-27, on the same spectra, so every contrast with `c0` is paired.
> One seed.
>
> - **The eval-mode gap narrows by 0.025-0.028 at matched epoch, and train fit rises.** At epoch
>   63 the gap is 0.158 against `c0`'s 0.183 on the probe sample (0.166 against 0.194 with the
>   full val split). Train rises by +0.003. No arm whose gap was measured narrowed it without
>   losing train fit: `k1` closed about 0.015 by fitting train 0.009 less.
> - **Half the narrowing is Lumos.** On the full val split, Lumos gains +0.054 and Elite +0.025.
>   `c0`'s extra Lumos gap (0.226 against Elite's 0.173) halves to 0.182 against 0.157.
> - **In `c0`, identity routes stood in for the missing instrument.** Only 7 of 12,668 train
>   molecules have spectra on both instruments, so a molecule's identity gives its instrument.
>   - On train, `c0` fits Lumos spectra as well as Elite ones but leans harder on the set message
>     (+0.028) and on global attention (+0.042) to do it. In `i1` that excess is 0.004-0.010.
>   - On val, `c0`'s only instrument cue was the NCE value. It scores Lumos spectra at 0.60 where
>     the NCE value is mostly Elite and at 0.76 where it is mostly Lumos.
> - **Every train-specific route weakens except the boundary block.** The set message falls
>   from a damage ratio of 4.20 to 2.44, and global attention from 2.53 to 2.23. The boundary
>   block holds.
> - **Most of the lookup remains.** The gap is still 0.158-0.166. Train damage under each route
>   is still nearly flat across novelty, and val still uses the routes only on close analogues.
> - **By NCE, from predicted spectra: the gain and the narrower gap hold in every CE bin.** On
>   Lumos val the precursor-fraction bias falls from +0.092 to +0.020, so the under-fragmentation
>   is gone above NCE 10. New: Elite val over-predicts the precursor at NCE ≤ 35 (+0.035 to
>   +0.087). Train shows no such bias.
> - **Follow-up: no Lumos-specific energy or shape error is left.** The leftover precursor bias
>   is shared by both instruments and grows with novelty. A cross-fitted precursor rescale
>   gains nothing on `i1` (+0.0002 on Lumos, against +0.0117 on `c0`). Matched on the measured
>   extent instead of NCE, Lumos's cosine gap is +0.005 [−0.007, +0.018].

## Question

`i1` beat `n4` by +0.021 mid-run, more than closing `c0`'s Lumos under-fragmentation alone would
give ([instrument finding §8](../nist_orbitrap_instrument_ce_scales_2026_09_28.md#8-training-with-the-instrument-input-i1-and-i2-running-read-2026-09-28)).
Does the instrument input change what the model memorizes, or does it only lift val? The
memorization instruments are:

- the eval-mode train/val gap;
- the route map, which corrupts one route at inference and compares train and val damage
  (route map (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_route_map_2026_09_24.md`),
  set-message probe on `c0` (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md`));
- the novelty bins.

## Method

- **Probe.** [`debug_pooled_embedding_memorization.py`](../../../scripts/debug/masster_memorization/debug_pooled_embedding_memorization.py),
  eval mode, 60 batches of 512 per split. Scopes are `setmsg`, `set`, `cc`, `boundary` and
  `gsa`, with modes `molmean` and `shuffle`. These are `c0`'s settings, and they cover 30,720
  train and 30,720 val spectra.
  - The spectrum sets match `c0`'s exactly on both splits (checked), so the `i1` − `c0` deltas
    are paired.
  - The relabel changed no split membership or row order.
- **Checkpoints.**
  - `model-epoch=059.ckpt` (best): full route table and per-spectrum dump, 23 min.
  - `model-epoch=063.ckpt`: plain gap only, to match `c0`'s recorded epoch-63 gap, 7 min.
  - Both md5-checked against W&B.
  - Config: [`i1_inst_ob_obf.yml`](../../../config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml),
    on `spec_df.pkl` md5 `bdc009bb…`.
- **Full val split.** `i1`@59 also ran on all 52,020 val spectra, 5 min.
  - The probe's val sample is the `spec_id`-ordered prefix
    ([audit, correction 8](masster_memorization_audits.md#corrections-to-the-record)).
  - It turns out to be 90% Elite, against 74% for the split.
- **`c0` references.**
  - The stored `c0`@42 probe (`output/allocator_route_nl_2026_09_26/`).
  - `c0`@63 from the [plan's eval-mode gap table](../../history/masster_memorization_route_plan_log.md#eval-mode-trainval-gap-on-the-step-1-checkpoints-2026-09-25).
  - `c0`@42's full-split val cosine by instrument from
    [instrument finding §5](../nist_orbitrap_instrument_ce_scales_2026_09_28.md#5-masster-c0-on-the-two-instruments).
    There, `OB` = Elite + Velos.
  - f0@61 (`3bej6fq7`, same recipe) is the later-epoch reference.
- **Reads.**
  - The damage ratio is train drop over val drop.
  - Excess gap closed is the gap the cc pools leave at the same val level, minus the
    condition's gap: linear through `none`, `cc_molmean` and `cc_shuffle`, as in the route map.
  - Instrument is the relabelled `inst_type`.
  - Novelty is the maximum Morgan-r2 Tanimoto to another train molecule (train) or to any train
    molecule (val).
  - Uncertainties are clustered by molecule.
- **Not run: `n4`, `i1`'s same-code control; the scope was `i1`.** It was trained on `FT` labels,
  but it can still run on the relabelled data. The dropout-view probe's
  `scripts/debug/masster_memorization/patch_inst_embedding.py` copies a one-class
  checkpoint's instrument row into two equal rows, which keeps its `FT` predictions. So `i1` −
  `c0` includes `n4`'s `fourier` CE table (`n4` − `c0` = +0.0053 at the tail).

## Result: eval-mode gap

Cosine is `cos_sim_0.01`. Train is 60 random batches. "Probe" val is the first 60 val batches;
"full split" is W&B's `val_cos_sim_0.01_epoch/mean` at that epoch. The probe reproduces it
exactly on the full split (0.7622 at epoch 59).

| checkpoint | epoch | train | val, probe | val, full split | gap, probe | gap, full split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `i1` | 59 | 0.9258 | 0.7696 | 0.7622 | 0.1562 | 0.1637 |
| `i1` | 63 | 0.9253 | 0.7669 | 0.7590 | **0.1584** | **0.1662** |
| `c0` | 42 | 0.9162 | 0.7402 | 0.7292 | 0.1760 | 0.1870 |
| `c0` | 63 | 0.9222 | 0.7388 | 0.7279 | 0.1834 | 0.1943 |

- **Epoch 63 against epoch 63.**
  - The gap is −0.0250 on the probe and −0.0281 on the full split.
  - Train is +0.0031.
  - Val is +0.0281 on the probe and +0.0311 on the full split.
- **Earlier arms on the same probe sample:**
  - `k1`@61: gap 0.1665, train 0.9132. It closes about 0.015 by lowering train fit.
  - `u2` (EMA)@56: 0.1808. It lifts both splits.
  - `r2`@57: 0.1765.
  - `m1`@56: 0.0919, by underfitting (train 0.814).
- **W&B tails, epochs 54-63, full split.**
  - `i1` 0.7607, `n4` 0.7329, `c0` 0.7276.
  - `i1` is +0.0278 over `n4` (6.6 times the 0.0042 seed spread) and above it at 64 of 64 epochs.
  - Against `n4`: train-mode primary loss −0.008, val loss −0.092, coverage +0.0002.

## Result: routes

Damage ratio, then excess gap closed, on the probe sample:

| route | `i1`@59 | `c0`@42 | f0@61 |
| --- | --- | --- | --- |
| set message, `molmean` | **2.44**, 0.034 | 4.20, 0.050 | 3.37, 0.050 |
| set message, `popmean` | **2.14**, 0.041 | 3.11, 0.064 | 2.82, 0.065 |
| `set_drop` (50% of keys) | 1.83, 0.008 | 2.11, 0.013 | 1.75, 0.012 |
| global attention, `popmean` | **2.23**, 0.098 | 2.53, 0.121 | 2.58, 0.123 |
| global attention, `molmean` | 2.13, 0.036 | 2.42, 0.062 | — |
| global attention, `xmol` | 1.98, 0.110 | 2.16, 0.126 | — |
| boundary block, `molmean` | 1.40, 0.085 | 1.45, 0.080 | 1.47, 0.085 |
| cc pools (the reference) | 1.15-1.19 | 1.21-1.24 | 1.21-1.25 |

f0's global-attention and boundary values are from the route map, on the same batches.

- **The set message is the route that moves most.**
  - Its train drop falls: 0.064 against 0.070 on `c0` and 0.078 on f0.
  - Its val drop rises: 0.026 against 0.017 and 0.023.
  - Val now uses it more, and train depends on it less.
- **Global attention writes less of the key.** Train drop is 0.201 against 0.231, at equal val
  drop (0.090 against 0.092).
- **The boundary block is unchanged.** Its excess of 0.085 is f0's.
- Coverage is identical across conditions (`wrecall` 0.947 train, 0.945 val).

## Result: by instrument

Val is the full split; train is the probe's random batches. ± is a 95% interval clustered by
molecule.

| | Elite (`OB`) | Lumos (`OB_F`) |
| --- | ---: | ---: |
| val spectra / molecules | 38,310 / 3,054 | 13,710 / 1,173 |
| `i1`@59 val | 0.7693 ± 0.0069 | 0.7423 ± 0.0109 |
| `c0`@42 val | 0.7439 | 0.6880 |
| val gain | +0.0254 | **+0.0543** |
| train, `i1` / `c0` | 0.9264 / 0.9170 | 0.9242 / 0.9142 |
| gap, `i1` / `c0` | 0.157 / 0.173 | **0.182 / 0.226** |

- **Lumos's excess gap halves.** It falls from 0.053 to 0.025.
  - Weighted by val share, Elite's gap change (−0.016 × 74%) and Lumos's (−0.044 × 26%) each
    give about −0.012.
  - Lumos val still trails Elite by 0.027. Within novelty bins the lag is 0.004-0.023 in five of
    six, because Lumos val molecules are more novel (median nearest-train Tanimoto 0.548 against
    0.607).
- **Train fits both instruments alike in both checkpoints.** `i1` gains +0.010 on each. Part of
  that is epoch 59 against 42: `c0` itself gains +0.006 from epoch 42 to 63.

Train damage by instrument on the probe sample, where train cosine is equal across instruments:

| condition | `c0` Elite | `c0` Lumos | `i1` Elite | `i1` Lumos |
| --- | ---: | ---: | ---: | ---: |
| set message, `molmean` | 0.063 | **0.091** | 0.063 | 0.067 |
| global attention, `popmean` | 0.220 | **0.262** | 0.198 | 0.208 |
| boundary block, `molmean` | 0.506 | 0.513 | 0.509 | 0.536 |
| cc pools, `molmean` | 0.407 | 0.371 | 0.381 | 0.377 |

- **In `c0` the identity routes carried more of a Lumos train spectrum's fit** than of an Elite
  one: +0.028 (set message) and +0.042 (global attention). The pools carried less. In `i1` the
  excess is +0.004 and +0.010.
- **Val damage ratios by instrument are not read.** `c0`'s Lumos val is low, so every
  corruption costs it less, which inflates the ratio.

### The NCE value was `c0`'s instrument cue on val

Paired on the probe sample, by the Lumos share of the spectrum's NCE value among train spectra:

| val spectra | NCE value's Lumos share in train | `c0`@42 | `i1`@59 | gain |
| --- | --- | ---: | ---: | ---: |
| Lumos, 635 | ≤ 0.2 | 0.595-0.597 | 0.718-0.767 | +0.12 to +0.17 |
| Lumos, 1,402 | 0.2-0.8 | 0.702-0.704 | 0.768-0.792 | +0.064 to +0.089 |
| Lumos, 1,027 | > 0.8 | 0.761 | 0.809 | +0.049 |
| Elite, 26,993 | ≤ 0.5 | 0.709-0.757 | 0.732-0.785 | +0.019 to +0.028 |
| Elite, 663 | 0.5-0.8 | 0.6838 | 0.7225 | +0.0387 |

- **Lumos.** `c0` predicted Lumos val spectra well only where the NCE value itself named the
  instrument. This is the per-NCE code carrying the instrument that `i2` − `i1` showed.
- **Elite.** The gain is flat in the NCE value, so it is not ladder confusion.
  - If `n4`'s +0.0053 over `c0` sat entirely on Elite, the `fourier` table would account for
    about 0.007 of it.
  - The rest has no mechanism isolated here.

## Result: by novelty

Paired `i1`@59 − `c0`@42 on the probe sample, by nearest-train Tanimoto. Standard errors are
0.003-0.007 on val and 0.001 on train.

| split | ≤ 0.4 | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | > 0.8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val, all | +0.028 | +0.027 | **+0.045** | +0.034 | +0.017 | +0.018 |
| val, Elite only | +0.025 | +0.023 | +0.032 | +0.029 | +0.015 | +0.014 |
| train | +0.012 | +0.009 | +0.009 | +0.008 | +0.010 | +0.010 |

- **The gain is largest on novel molecules.** Up to Tanimoto 0.7 it is +0.027 to +0.045; above,
  +0.017 to +0.018. The same shape holds within Elite, so it is not the instrument mix.
- **The lookup signature persists in `i1`.**
  - Train damage is nearly flat across novelty: set message 0.063-0.066, global attention 0.19-0.23.
  - Val damage rises with similarity: set message 0.008 → 0.044, global attention 0.045 →
    0.120. `c0`'s rose −0.008 → 0.032 and 0.038 → 0.142.

## Result: by NCE, from predicted spectra

**Method.**

- `collect_masster_preds.py`, unchanged, saved `i1`@59's predicted and measured spectra, as
  it had for `c0`@42. It covers every val spectrum and a 4,000-molecule train sample (seed 0,
  the same molecules as `c0`'s).
- `ladder_bias.py` and `nist20_raw_labels.py`, given the new run-tag argument, compute the
  per-spectrum fragmentation extent and read it by raw instrument label.
  - The precursor fraction is the intensity within max(0.01 Da, 10 ppm) of the precursor m/z.
  - Its bias is predicted minus measured, so a positive bias means under-fragmented.
- `plot_ladder_bias_compare.py`, adapted from `plot_ladder_bias.py`, draws the figure. Each
  point needs 100 spectra.
- Everything is in `output/ev_ladder_census_2026_09_28/`. Velos (1,446 val spectra) is left out.

The `i1` vs `c0` per-NCE figure (`masster_i1_vs_c0_elite_vs_lumos_per_nce.png`) was never
committed, and `output/ev_ladder_census_2026_09_28/` no longer exists; only the whole-split
table below remains.

On the whole val split, with 95% molecule-bootstrap intervals for `i1`:

| | `c0` Elite | `c0` Lumos | `i1` Elite | `i1` Lumos |
| --- | ---: | ---: | ---: | ---: |
| val cosine | 0.7450 | 0.6880 | 0.7699 [0.7627, 0.7765] | 0.7423 [0.7317, 0.7526] |
| val precursor-fraction bias | −0.0008 | **+0.0923** | +0.0233 [+0.0204, +0.0265] | **+0.0196** [+0.0163, +0.0233] |
| val centroid bias | −0.0095 | +0.0515 | +0.0092 | +0.0082 |
| train precursor-fraction bias | +0.0066 | +0.0055 | +0.0003 | +0.0019 |

The Elite − Lumos differences are matched on NCE bin × novelty bin × m/z quartile, with all
Lumos weight covered:

- cosine: +0.0525 on `c0`, +0.0240 on `i1`;
- precursor-fraction bias: −0.0905 on `c0`, +0.0113 on `i1`.

By CE bin (`ladder_bias.py`'s bins), `c0` → `i1`. The gap is train minus val cosine.

| NCE | Elite val | Elite gap | Elite precursor bias | Lumos val | Lumos gap | Lumos precursor bias |
| --- | --- | --- | --- | --- | --- | --- |
| ≤ 10 | 0.876 → 0.894 | 0.052 → 0.040 | +0.039 → **+0.087** | 0.795 → 0.838 | 0.151 → 0.116 | +0.215 → +0.099 |
| 10-20 | 0.792 → 0.819 | 0.133 → 0.112 | −0.005 → **+0.054** | 0.746 → 0.798 | 0.191 → 0.149 | +0.200 → +0.031 |
| 20-35 | 0.765 → 0.790 | 0.168 → 0.148 | −0.014 → +0.035 | 0.669 → 0.741 | 0.245 → 0.183 | +0.169 → +0.022 |
| 35-60 | 0.745 → 0.764 | 0.180 → 0.167 | −0.001 → +0.019 | 0.642 → 0.695 | 0.258 → 0.214 | +0.077 → +0.004 |
| 60-100 | 0.702 → 0.724 | 0.195 → 0.184 | +0.002 → +0.007 | 0.661 → 0.700 | 0.245 → 0.220 | +0.011 → +0.000 |
| > 100 | 0.703 → 0.739 | 0.186 → 0.168 | −0.000 → +0.000 | 0.678 → 0.758 | 0.240 → 0.178 | +0.001 → −0.000 |

Each bin holds 2,116-10,285 Elite and 1,565-4,134 Lumos val spectra.

- **Every bin gains, and every bin's gap narrows, on both instruments.** Lumos gains +0.04 to
  +0.08 and Elite +0.02 to +0.04.
- **Lumos's under-fragmentation is gone above NCE 10.** The bias is +0.022 to +0.031 at NCE
  10-35, where `c0` had +0.17 to +0.20. At NCE ≤ 10 it halves, to +0.099.
- **New: Elite val spectra are predicted under-fragmented at low NCE.**
  - At NCE ≤ 10 the predicted precursor fraction is 0.793 against a measured 0.706. At 10-20
    it is 0.556 against 0.502. `c0` was close to calibrated there (0.744 and 0.497).
  - On train, `i1`'s Elite bias in the same bins is −0.017 to −0.002. So it is val-only, and
    cosine still rises in those bins.
  - Both instruments now carry about the same small val-only bias (+0.02 overall). Whether it
    comes from `n4`'s `fourier` table or from the instrument input needs an `n4` run.
- **§8's pass criterion.** It asked that the Lumos bias fall well below +0.09 with no Elite
  loss.
  - The Lumos bias passes: +0.020.
  - Elite cosine rises, +0.025.
  - Elite's precursor bias rises from −0.001 to +0.023.

### Is an energy or shape error left? (follow-up, 2026-09-28)

This uses the same predicted spectra, paired per spectrum, with the same instrument labels and
novelty. Scripts and logs are in `output/ev_ladder_census_2026_09_28/i1_energy_shape/`.

- **No Lumos-specific energy error is left.**
  - Matched on NCE bin × novelty × m/z, Lumos's precursor bias is now 0.011 below Elite's.
    On `c0` it was 0.090 above.
  - The Lumos energy offset depends on mass
    ([instrument finding §3.2](../nist_orbitrap_instrument_ce_scales_2026_09_28.md#32-energy-how-much-and-in-what-form)),
    and so did `c0`'s bias.
    - At NCE ≤ 10, below m/z 300 / 300-450 / above 450, it was +0.19 / +0.27 / +0.27.
    - `i1` has +0.09 / +0.13 / +0.07 there, against Elite's +0.08 / +0.11 / +0.09.
    - Above NCE 10, Lumos is at or below Elite in every m/z bin.
- **No Lumos-specific shape error shows in these readouts.**
  - Intensity below m/z 100: bias 0.000 on Elite and −0.003 on Lumos. `c0` had +0.020 and
    −0.051.
  - Intensity below half the precursor m/z: bias −0.007 and −0.011 (+0.003 matched).
- **The leftover precursor bias costs no cosine.** The test scales the predicted precursor peaks
  by a factor fitted per instrument × NCE bin on half the molecules and scores the other half
  (2-fold by molecule, 5 splits). Adding novelty bins to the fit changes nothing.

  | cross-fitted cosine gain | `c0` Elite | `c0` Lumos | `i1` Elite | `i1` Lumos |
  | --- | ---: | ---: | ---: | ---: |
  | precursor rescale | +0.0017 | +0.0117 | +0.0000 | +0.0002 |

  - `c0`'s fitted factor cut the Lumos precursor to 0.2-0.4×. `i1`'s factors at NCE ≤ 35 are
    1.1-1.35×, so cosine would take more precursor, not less. The bias is in the mean precursor
    fraction, and per-spectrum cosine does not reward removing it.
  - It grows with novelty on both instruments. Elite goes from +0.053 at Tanimoto ≤ 0.4 to
    +0.007 at 0.7-0.85 (`c0`: +0.016 to −0.008), and Lumos from +0.035 to 0.000. Train is
    +0.000 and +0.002.
- **Lumos's remaining cosine gap comes from its targets' extent, not from the model.**
  - At a given NCE, Lumos spectra are more fragmented: measured precursor fraction at NCE 20 is
    0.30 against Elite's 0.46, and at NCE 35 it is 0.10 against 0.23.
  - More fragmented spectra score lower on both instruments.
  - Matched on the measured extent instead of NCE, the gap is within noise on `i1`. Extent here
    is the precursor-fraction bin, with fully fragmented spectra split by measured-centroid
    tertile, crossed with novelty × m/z; all Lumos weight is covered.

  | Elite − Lumos val cosine | `c0` | `i1` |
  | --- | ---: | ---: |
  | matched NCE bin | +0.0525 [+0.038, +0.065] | +0.0240 [+0.012, +0.036] |
  | matched measured extent | +0.0326 [+0.020, +0.046] | **+0.0053** [−0.007, +0.018] |

  - At equal extent, `c0` still lost 0.033 on Lumos. `i1` does not, to within ±0.012.
- **Limits.**
  - Matching on measured extent conditions on the target.
  - The rescale moves only the precursor peaks, so it bounds a full extent correction from
    below.
  - The rescale uses a reimplemented `cos_sim_0.01`. It reproduces the logged value within
    10⁻³ on 98% of spectra; the rest are 0.01 Da bin-edge cases, the same spectra in every
    condition.

## The probe's val sample

The first 60 val batches hold 90% Elite spectra, against 74% on the split, and they are less
novel: median nearest-train Tanimoto 0.613, against 0.556 for the rest of val.

| | `i1` prefix | `i1` full split | `c0` prefix | `c0` full split |
| --- | ---: | ---: | ---: | ---: |
| Elite | 0.7684 | 0.7693 | 0.7447 | 0.7439 |
| Lumos | 0.7813 | 0.7423 | 0.7002 | 0.6880 |

- **The Elite prefix is representative; the Lumos prefix is easier.**
- **The prefix's overall bias is mostly the instrument mix.** It is +0.011 on `c0` and +0.0074
  on `i1`, the smaller because Lumos lags less.
- **Prefix Lumos reads overstate the full split.** The paired prefix Lumos gain is +0.081,
  against +0.054 on the full split; the NCE-share table is on the prefix too.
- Post-stratifying the prefix to the split's instrument mix recovers `c0`'s full-split val to
  within 0.004 (0.7329 against 0.7292), but `i1`'s only to within 0.010 (0.7718 against
  0.7622), because its Lumos prefix is the easier part.
- This answers the audit's next check 5, on `i1`.

## Gates

These are the [memorization plan's five](../../plans/masster/masster_memorization_route_plan.md#objective-and-acceptance).
`i1` is an input change, not a regularizer; the gates are read so it can be compared with the
arms.

1. **Validation: pass.** +0.0278 over `n4` on one seed.
2. **Ranking: not run.**
3. **No rerouting: pass, against `c0`.** The gap falls. No route's excess grows beyond the
   control's: the set message and global attention fall, and the boundary block holds at f0's
   value.
4. **Novelty: pass.** The gain is largest at Tanimoto ≤ 0.7, within Elite too.
5. **Coverage: pass.** +0.0002.

## Reading

- **This is the first change on the unmerged lineage that narrows the eval-mode gap while train
  fit rises.** Earlier narrowing came from lower train fit (`k1`, `m1`), or did not happen (EMA,
  route removals).
- **Part of what the probes read as memorization was a missing input.**
  - A train molecule's identity names its instrument. `c0` used that to fit Lumos train spectra
    as well as Elite ones, leaning harder on the identity routes.
  - A val molecule had only its NCE value to go on.
  - With the instrument as an input, the identity routes no longer carry it, and Lumos transfers.
- **The per-molecule lookup is otherwise intact.** About 86% of `c0`'s gap remains at matched
  epoch, and every route's train damage is still nearly flat across novelty.
  - The levers that act on that lookup (`k1`, EMA) are what `i3` stacks on `i1`
    ([plan](../../history/masster_memorization_route_plan_log.md#stacking-on-the-instrument-input-i3-and-i4-built-2026-09-28-not-submitted)).
  - `k1` and EMA were measured on the `FT` labels, so whether they add on `i1` is open.
- **Earlier probe reads carry the instrument proxy.** Route maps and gap reads on `FT`-label
  checkpoints mix the instrument cue into the lookup. Their val side is 90% Elite.
- **Lumos still trails Elite by 0.027 on val** (+0.024 matched on NCE bin), though its energy
  bias is gone. Matched on the measured extent, the gap is +0.005, within noise. So what remains
  is that Lumos spectra at a given NCE are more fragmented, and more fragmented spectra are
  harder on either instrument. The spectra still differ beyond energy
  ([instrument finding §3.3](../nist_orbitrap_instrument_ce_scales_2026_09_28.md)), but `i1`
  shows no Lumos-specific error on that difference in these readouts.

## Limits

- **One seed per checkpoint.**
- **The paired reference is `c0`, not `n4`.** So `i1` − `c0` includes the `fourier` table.
- **The route table compares different epochs** (`i1`@59, `c0`@42). f0@61 agrees with `c0`
  wherever it was read.
- **Several reads use the probe sample only:** the per-instrument route damage, the novelty
  bins and the NCE-share table. Its val prefix holds 285 Lumos molecules and is easier than the
  split.
- **Inference-time corruption shows what the trained model reads, not what training would do.**

## Artifacts

Everything is in `output/i1_memorization_2026_09_28/` (local, Git-ignored):

- `run_probe.sh`: the two probe runs.
- `run_fullval.sh`: the full-val run.
- `prep_meta.py`: writes `meta.pkl` (instrument, NCE, novelty per spectrum).
- `analyze_i1.py` and `analyze_i1_fullval.py`, with logs `analyze_i1_log.txt` and
  `analyze_i1_fullval_log.txt`.
- The probe CSVs and per-spectrum pickles (`i1_e059_*`, `i1_e063_*`, `i1_e059_fullval_*`).
- W&B epoch tables `wandb_*_epochs.csv`.

The per-NCE read is in `output/ev_ladder_census_2026_09_28/` (local, Git-ignored):

- `i1_val.pkl` and `i1_train.pkl`: predicted spectra, logs `collect_i1_*.log`.
- `i1_*_ladder_bias*.pkl`, with logs `i1_ladder_bias_log.txt` and `i1_raw_labels_log.txt`.
- `plot_ladder_bias_compare.py`, which writes `c0_vs_i1_ladder_bias.png`, the plotted values in
  `c0_vs_i1_per_nce.csv` and the per-bin log `c0_vs_i1_ladder_bias_log.txt`.
- `i1_energy_shape/` holds the energy-or-shape follow-up. It has `i1_lumos_bias.py`,
  `i1_train_bias.py`, `prec_scale_terms.py`, `prec_scale_read.py` and `matched_extent_gap.py`,
  each with a `_log.txt`, plus the paired per-spectrum pickles. Run them from the repository root
  with that folder as the argument.

Checkpoints are in `saved_ckpts/i1_inst_ob_obf_i50ig7rf/ckpt/`.
