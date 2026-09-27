# Would a QET breakdown-curve head change MASSter's predictions, and can one candidate set serve every energy?

> **Status:** FINDING · **Updated:** 2026-09-26
> Two offline gates for a MASSter intensity head built on the
> [breakdown-curve form](../breakdown_curve_form_fit_2026_09_26.md), `softmax(a_i − e_i·β(NCE))`,
> run once per molecule, plus what they exposed about the NCE input. Checkpoint
> `u2_ema999_c8hh9j1b.ckpt` (EMA weights), unmerged `[M+H]+`/FT, all 52,020 val spectra and
> 24,726 spectra of 2,000 train molecules; no training.
>
> - **The model's CE series already sit almost on the form.** `u2`'s own predictions were
>   projected, per molecule, onto the one- or two-component form, fitted to the predictions and
>   never the targets. The projection moves them by 0.8% and 0.35% (cosine to the original 0.9918
>   and 0.9965). It gains **+0.0019 val cosine** (SE 0.0002, clustered by molecule), and +0.0023
>   on top of `P^0.75`, so it is not flattening. On train it loses 0.0009, where the model's series
>   are exactly as consistent as measured spectra.
> - **So the output law has little to correct.** A head whose only effect was to enforce it would
>   be worth about +0.002, under the +0.003 gate and the 0.0042 seed spread. This extends the
>   form-fit finding's precursor-only projection (+0.0006) to every channel. The val error is at
>   the molecule level, not noise from one energy to the next.
> - **One CE-blind candidate set per molecule is feasible.** The fragments `u2` keeps differ by
>   energy: 109 are kept at every energy, and the union is a median 392 fragments, 1.65 times one
>   energy's set. The extra ones add little. The 256 best, ranked by their top proposal
>   log-probability, cover 0.9422 of target intensity against 0.9402 for each energy's own set
>   (+0.0020, SE 0.0003), and 384 cover 0.9471 (+0.0069).
>   - Ranked by mean proposal probability, which a proposal trained on the merged target should
>     approach, the set covers the same and loses no energy
>     ([section](#what-a-shared-proposal-would-rank-2026-09-26)).
>   - Most of the union is same-m/z isomers. Skipping them, 128 candidates cover 0.9428, more
>     than each energy's own 256.
> - **The NCE input is categorical, not an energy axis.** With `ce_insert_type: embed` and
>   `int_embedder: abs-sines`, which 245 of the 262 MASSter configs use, each integer NCE gets a
>   fixed binary-like code.
>   - Codes of neighbouring values differ by up to 1.0.
>   - NCE x and 256 − x get the same code, so 76 and 180, both in the training data, are
>     indistinguishable.
>   - Feeding NCE 170 or 190, which training never saw, to spectra measured at 175-185 drops
>     cosine to 0.22-0.42.
>   - Rarely trained high values gain most from projection (+0.03 to +0.07 at NCE 165-195).
>   - The same table serves FraGNNet and the other models (749 configs in all), and it also
>     aliases the ACE path: 30 and 98 eV collide at the template `ace_max: 100`.
> - **Fixed by a new option, `ce_insert_type: qet`**, in both CE code paths, with existing configs
>   and checkpoints unchanged. On the train CE series with seven NCE values held out, a
>   standardized `1/(NCE + 200)` scalar predicted the unseen energies at 0.9845 channel cosine,
>   against 0.9480 for abs-sines and 0.9627 for 32 RBF bumps
>   ([section](#fix-a-smooth-energy-input-2026-09-26)). Arm `n1` (`c0` plus `qet`, `jxczo2kf`)
>   trails `c0` by 0.0137 at epochs 24-33 of 64 (mid-run; projected tail −0.012). Its train loss
>   equals `c0`'s and only val loss is worse, so the smooth input costs generalization, not fit,
>   as the ladder reading predicts. `n2` (`qet_mlp`) and `n3` (`id`) are running
>   ([section](#validation-arm-n1)).
>
> The concrete uses of QET here are a smooth, monotone energy input and a once-per-molecule pass.
> Enforcing the law on `u2`'s output would change little, because `u2` already obeys it on the
> energies it was trained on.

## Question

The form fits measured CE series (0.995 channel cosine), so a head could restrict MASSter's CE
dependence to it: a CE-blind candidate token emits `(a_i, e_i)`, or three numbers for two
components, and each energy costs one softmax. Two things decide whether that is worth
building:

1. **Does the constraint bind?** If `u2`'s predicted CE series already obey the form, enforcing
   it changes little. If they do not, projecting them onto it shows whether the violation is
   error (projection helps) or information (it hurts).
2. **Can one candidate set serve every energy?** A once-per-molecule pass needs a CE-blind set.
   `u2` selects up to 256 fragments per spectrum from CE-conditioned bond scores.

## Method

Scripts: `output/qet_projection_2026_09_26/` (local, Git-ignored), run from the repository root.

- `collect.py` runs inference, and `project.py` and `coverage.py` analyze its output.
- `analyze.py` prints the tables, and `mismatch_diag.py` checks candidate channels.
- `nce_sweep.py` and `nce180_check.py` probe the NCE input.
- `fit_m2g_curves.py` and `encoding_probe.py` run from that directory.

- **Data.** `config/masster/mh_ft_ranking_transfer_2026_09_23/u2_ema999.yml`, with
  `data/split/nist20_inchikey`. Val: all 52,020 spectra of 4,222 molecules. Train: every spectrum
  of 2,000 molecules drawn with seed 0, 24,726 spectra. A first train run on 4,222 molecules was
  killed at batch 336 of 407 when another process filled memory.
- **Collection.** `inference_step` with `return_explainability_tensors`, eval batch 128. Per
  spectrum it keeps the evaluated prediction and target (the arrays `cos_sim_0.01` bins), the
  model's `cos_sim_0.01`, and the kept fragments. A fragment's identity is its atom set, so one
  kept at several energies is one fragment. The mean `cos_sim_0.01` is 0.7394, as in the
  [precursor-survival oracle](../breakdown_curve_form_fit_2026_09_26.md#precursor-survival-ceiling-on-val-2026-09-26).
- **Channels.** Per molecule, one channel per exact predicted m/z across all its energies. The
  model's peaks sit at exact formula masses, so the same ion has the same m/z at every energy.
  Rebuilding each prediction on these channels leaves its score unchanged.
- **Forms, fitted to the predictions.** The same per-spectrum cross-entropy as the form fit, with
  the model's predicted spectrum as the target:

  | form | logit of channel i at energy k | curves |
  | --- | --- | --- |
  | M0 | `a_i` (CE-blind mean spectrum) | none |
  | M1g | `a_i − e_i·β(NCE_k)` | β from the train-target M1g fit, fixed |
  | M2g | `a_i + b_i·u(NCE_k) + c_i·v(NCE_k)` | u, v from the train-target M2g fit, fixed |
  | M1n | `a_i − e_i·β_n` | β free per distinct NCE value, fitted to the predictions |

  - The curves come from train targets only. `fit_m2g_curves.py` refits M2g and reproduces the
    form fit's 0.9985 / 0.9919 / 0.0076.
  - With the curves fixed, molecules are independent, so they are fitted in chunks of about 500.
  - M1n tests whether the fixed curve misplaces some energies.
- **Scoring.** Each projected spectrum is binned and scored exactly as `cos_sim_0.01` (0.01 Da,
  sum within bins). Re-scoring the unprojected prediction reproduces the model's per-spectrum
  value to within 1.1e-6. Two controls:
  - `P^0.75` variants raise each spectrum to 0.75 before binning. The
    [ranking decomposition](masster_ranking_decomposition_2026_09_23.md) found that flattening
    worth +0.007; a projection gain that is only flattening would vanish on top of it.
  - Standard errors are clustered by molecule, because a projection acts on a molecule's spectra
    jointly.
- **Candidate union.** Coverage is the fraction of target intensity within tolerance of any
  candidate ion m/z: `_candidate_weighted_recall`, 10 ppm with a 200 Da floor (the FT setting).
  - `own`: the fragments kept for that spectrum, the set `u2` trains and predicts with.
  - `union`: every fragment kept at any of the molecule's energies.
  - `shared_N`: the union capped at N, ranked CE-blind by the best proposal log-probability the
    fragment reached at any energy. That ranking uses the per-energy proposals, so it shows that
    a good shared set exists, not that a CE-blind proposal learns it.
- **NCE input.** The context encoder's NCE features were read from the checkpoint. Val spectra
  were re-scored with their NCE input replaced (all other inputs unchanged).

## Result

### Projection onto the form

Val, 52,020 spectra, 4,222 molecules:

| variant | cosine | change | SE (by molecule) | cosine to the prediction |
| --- | ---: | ---: | ---: | ---: |
| prediction | 0.7394 | | | 1 |
| M0, CE-blind mean spectrum | 0.5370 | −0.2024 | 0.0014 | 0.6805 |
| M1g, one component | 0.7413 | +0.0019 | 0.0002 | 0.9918 |
| M2g, two components | 0.7413 | +0.0019 | 0.0001 | 0.9965 |
| M1n, β free per NCE value | 0.7411 | +0.0017 | 0.0002 | 0.9924 |
| `P^0.75` | 0.7461 | +0.0067 | 0.0003 | |
| M1g, then `P^0.75` | 0.7484 | +0.0023 over `P^0.75` | 0.0001 | |
| M2g, then `P^0.75` | 0.7480 | +0.0019 over `P^0.75` | 0.0001 | |
| M1n, then `P^0.75` | 0.7482 | +0.0021 over `P^0.75` | 0.0001 | |

- **The projection barely moves the prediction.** One component changes it by 0.8% (cosine
  0.9918), two by 0.35%. Measured train spectra sit at 0.9950 and 0.9985 of their own projection
  ([form fit](../breakdown_curve_form_fit_2026_09_26.md#in-sample); subformula channels, so only
  roughly comparable). On val `u2`'s series are slightly less QET-consistent than measured ones;
  on train they match ([below](#train-sample)).
- **What it removes is slightly more error than signal.** +0.0019, with 49% of spectra helped and
  44-47% hurt. It survives `P^0.75` (+0.0023), so it is not a temperature effect.
- **Freeing β per NCE value changes nothing** (+0.0017), so the fixed curve is not misplacing
  energies.
- **CE matters to the model.** Its CE-blind mean spectrum loses 0.20.

By where the spectrum sits (val, change from the prediction):

| slice | spectra | prediction | M1g | M2g |
| --- | ---: | ---: | ---: | ---: |
| lowest energy of its molecule | 4,467 | 0.8824 | +0.0005 | +0.0001 |
| interior energy | 43,318 | 0.7271 | +0.0023 | +0.0023 |
| highest energy | 4,235 | 0.7137 | −0.0003 | −0.0008 |
| nearest-train Tanimoto ≤ 0.4 | 6,516 | 0.6121 | +0.0027 | +0.0021 |
| 0.4-0.6 | 21,036 | 0.7034 | +0.0023 | +0.0021 |
| 0.6-0.8 | 20,156 | 0.7963 | +0.0017 | +0.0017 |
| > 0.8 | 4,312 | 0.8415 | −0.0002 | +0.0011 |

- **It helps at interior energies**, where neighbours on both sides constrain the curve, and not
  at a molecule's ends.
- **It helps novel molecules a little more**, +0.0027 at Tanimoto ≤ 0.4 against about 0 above
  0.8. That is still small against the 2-run ensemble's gains on the same novel molecules
  ([variance is epistemic](masster_variance_novelty_2026_09_24.md)).

### Where the gain sits: shared and rare NCE values

Val molecules were measured on two main NCE ladders:

- **Ladder A** (2,589 molecules, prediction 0.7549): 10, 15, 20, 25, … 60, 75, 90, 110, 130, 160.
- **Ladder B** (1,103 molecules, 0.7028): 2, 5, 9, 14, 20, 27, 35, 44, 54, 65, 76, 88, 100, 115,
  130.

Here a molecule is on B if it has any of 9, 14, 27, 44, 54, 76, 88 or 100. It is on A if it has
none of those and none of 65, 80, 95 or 115. The other 530 molecules are mixed.

M1g change by NCE value and ladder (val; M1n in brackets):

| NCE | ladder A | ladder B |
| --- | ---: | ---: |
| 20 (shared) | +0.0047 (+0.0039) | +0.0107 (+0.0035) |
| 35 (shared) | +0.0078 (+0.0066) | +0.0140 (+0.0032) |
| 130 (shared) | +0.0107 (+0.0087) | +0.0086 (+0.0077) |
| A-only values 10-90 | −0.0014 to +0.0015 | |
| 160 (A's last) | −0.0099 | |
| B-only 9, 14, 27, 44, 54, 88, 100 | | −0.0063 to −0.0031 (−0.0024 to −0.0007) |
| 76 (B-only, shares its code with 180) | | +0.0023 (+0.0068) |
| 65, 115 | | +0.0189, +0.0059 |

- **The gain concentrates at the values both ladders use**, in both ladders' molecules. On
  ladder A, values the other ladders rarely use move by about 0.001.
- **One consistent reading.** The model conditions on the NCE value alone, and its NCE input is
  categorical ([next section](#the-nce-input-is-categorical)). A shared value then gets one
  prediction for two acquisition setups. If the setups deliver different effective energies at
  the same nominal NCE, that prediction fits neither population, and smoothing across a
  molecule's own neighbours corrects it. Freeing β per value does not remove the gain for
  ladder A, as that reading predicts. This is untested: a shift sweep cannot test it, because
  shifted inputs land on unrelated codes.
- **Rarely trained high values gain most**: +0.028 at NCE 165, +0.066 at 180, +0.035 at 185 and
  +0.039 at 195. These have 50, 164, 41 and 46 val spectra, and 116-537 train spectra each,
  against 2,000-10,000 for the common values.

### The NCE input is categorical

`u2`'s NCE path is a fixed feature table followed by a learned linear layer:

- **The table.** `FourierFeaturizerAbsoluteSines` with `max_count_int = nce_max + 1 = 201`
  gives 8 features, |sin(π·x/2^j)| for j = 1..8, with `requires_grad = False`. Its own docstring
  says it "will match the Binarizer quite closely".
- **What that implies:**
  - Every feature is unchanged under x → 256 − x, so the 201 integers have only 129 distinct
    codes. Among the 42 NCE values in training, 76 (2,942 train spectra) and 180 (537) collide
    exactly.
  - Neighbouring values are far apart: NCE 35's code differs from 34, 36 and 40 by 1.0 in some
    feature.
- **Who is affected.** MASSter reads `int_embedder` only for the NCE and ACE embeddings, and the
  template default is `abs-sines`.
  - 245 of the 262 MASSter configs set `ce_insert_type: embed`.
  - The other 17 (5 early merged baselines in `merged/baseline/` and 12 under `legacy/`)
    inherit the template's `id`, one standardized NCE scalar. No like-for-like comparison of the
    two is recorded.
  - The [feature audit](../../reference/masster_feature_set_audit.md) called this a learned
    embedding; only the linear layer is learned.
  - The shared `CEModel` (FraGNNet and the other models) builds its `embed` and
    `embed_clamped` paths the same way. Across `config/`, 532 configs set `embed` and 217 set
    `embed_clamped`. Only 4 set `int_embedder`, all to `abs-sines`.
  - The ACE path (collision energy in eV) uses the same table. At the template `ace_max: 100` it
    has 36 colliding pairs, including 30 and 98, 35 and 93, 40 and 88, and 50 and 78 eV. At
    `nce_max: 200` there are 72, from 56 and 200 through 127 and 129.

Val cosine with the NCE input replaced (each row is the spectra measured at that NCE):

| measured NCE (spectra) | own value | 76 | 170 | 175 | 185 | 190 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 180 (164) | 0.6668 | 0.6668 | 0.2741 | 0.6596 | 0.6322 | 0.2235 |
| 76 (937) | 0.6745 | 0.6745 | 0.6361 | 0.5045 | 0.6318 | 0.5531 |
| 175 (50) | 0.6681 | 0.5389 | 0.4225 | 0.6681 | 0.6063 | 0.2901 |
| 185 (41) | 0.5866 | 0.5080 | 0.3811 | 0.6483 | 0.5866 | 0.2596 |

- **The collision is exact:** inputs 76 and 180 give identical predictions. It costs the NCE-180
  val spectra nothing measurable: their own input is as good as 175 or 185, and no val molecule
  is measured at both.
- **An NCE the model never saw breaks it.** 170 and 190 are not in the training data. Fed to
  spectra measured at 175-185, they drop cosine to 0.22-0.42, against about 0.6-0.67 at the
  measured value.
- **A rarely trained value can be worse than its neighbour**: spectra measured at 185 (116 train
  spectra) score 0.5866 with their own input and 0.6483 with 175.
- **A shift sweep at shared values is uninterpretable for the same reason.** Shifts of ±2 to ±8
  around NCE 20, 35 and 130 moved cosine by up to 0.155, non-monotonically
  (`nce_sweep_log.txt`).

On NIST20 every val NCE also occurs in training, so this costs little here. It matters for any
dataset or instrument with other NCE values, and for QTOF collision energies in eV through the
ACE path, which uses the same featurizer.

### Other integer embeddings (2026-09-26)

Every other `get_embedder` table in `src/` was checked for the same aliasing. No config sets
`int_embedder_tight`, so FraGNNet's count features use the default 255-entry table. There
abs-sines aliases x with 256 − x, so a collision needs a value of 129 or more in the data.

| embedding | configs using it | value range in the data | aliased in practice |
| --- | ---: | --- | --- |
| NCE / ACE (`embed`, `embed_clamped`) | 749 | NCE 2-195; ACE up to 300 | **yes**; NCE 76 = 180, ACE 30 = 98 eV at `ace_max: 100`; fixed by `qet` |
| FraGNNet `base_formula`, node and edge | 233 + template | heavy-element counts, C ≤ 129 (H is zeroed, `frag_utils.py:1373`) | C = 129 in 1 NIST20 molecule |
| candidate formula (candidate-ion head, includes H) | 2 | H up to 223 | H ≥ 129 in 30 molecules, 878 spectra at precursor m/z ≤ 1500 |
| `neutral_loss_formula` | 1 | heavy-element counts | no |
| `depth` | 233 + template | 0-5 | no |
| `out_degree`, `in_degree` | 1 (debug) | up to 1,438 and 3,243 | **yes**; 0.52% and 0.17% of DAG nodes in 129-254, in 21.7% and 13.9% of molecules (1,500-molecule sample); 255 and up share one overflow token |
| `num_heavy_atoms` | 2 | up to 184 | 7 molecules at precursor m/z ≤ 1500 |
| `rdbe` (2×RDBE), `num_cut_bonds` | 1, 0 | small | no |
| candidate H-channel, CMF flag | 2 | tables of 9 and 2 | no |
| ICEBERG formula counts (own copy of the table) | baseline | element counts | as the candidate formula; kept as the original ICEBERG code |

MASSter's candidate features use `log1p` scalars, and MASSformer's degree embeddings are Graphormer
lookups over molecular-graph degrees. Adduct, instrument and fragmentation-mode embeddings are
categorical by nature. None of these share the problem.

- **Only the CE input was both aliased and in wide use.** It is fixed.
- **The DAG degree features alias hub nodes**, but only a debug config enables them. If they are
  turned on, they need a smooth encoding such as `log1p`, as MASSter uses, not the abs-sines
  table.
- **The count features keep a binary-like code**, like CE did. Their coverage is dense, so a
  value unseen in training is rare, unlike CE's 42 sparse values. No change is proposed without a
  training comparison.

### Train sample

2,000 train molecules, 24,726 spectra. The model scores 0.9290 on them.

| variant | train change | val change | train cosine to the prediction | val |
| --- | ---: | ---: | ---: | ---: |
| M1g | −0.0009 (SE 0.0001) | +0.0019 | 0.9955 | 0.9918 |
| M2g | +0.0007 (SE < 0.0001) | +0.0019 | 0.9987 | 0.9965 |
| M1n | −0.0008 (SE 0.0001) | +0.0017 | 0.9956 | 0.9924 |
| `P^0.75` | −0.0049 | +0.0067 | | |
| M1g, then `P^0.75` | −0.0003 over `P^0.75` | +0.0023 | | |

- **On train the model's series are exactly as QET-consistent as measured spectra**: 0.9955 and
  0.9987 against the targets' 0.9950 and 0.9985. There the off-form part is signal, and the
  one-component projection loses 0.0009.
- **On val they leave the form about twice as far** (1 − 0.9918 = 0.0082 against 0.0045), and the
  extra is mostly error, so projection gains.
- **This is a generalization signature in the CE dimension, and a small one.** Val minus train is
  +0.0028 for M1g and +0.0012 for M2g. It bounds what enforcing the law after training could
  recover at about +0.002.

### One candidate set for every energy

Val, distribution over the 4,222 molecules:

| | mean | p10 | median | p90 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fragments kept per spectrum (molecule mean) | 228 | 174 | 251 | 256 | 256 |
| fragments kept at any energy (union) | 414 | 187 | 392 | 650 | 994 |
| fragments kept at every energy | 109 | 41 | 97 | 202 | 247 |

Only 15.3% of molecules have a union of 256 or fewer.

| candidate set | coverage | change from `own` | SE (by molecule) | spectra losing > 0.01 |
| --- | ---: | ---: | ---: | ---: |
| `own` (per spectrum) | 0.9402 | | | |
| `union` | 0.9505 | +0.0104 | 0.0004 | 0.0% |
| `shared_128` | 0.9273 | −0.0129 | | |
| `shared_256` | 0.9422 | +0.0020 | 0.0003 | 5.2% |
| `shared_384` | 0.9471 | +0.0069 | 0.0003 | 1.3% |
| `shared_512` | 0.9490 | +0.0088 | | |

By NCE (val):

| NCE | `own` | `union` | `shared_256` | fragments kept per spectrum |
| --- | ---: | ---: | ---: | ---: |
| ≤ 20 | 0.9680 | 0.9712 | 0.9676 | 238 |
| 20-30 | 0.9643 | 0.9702 | 0.9634 | 239 |
| 30-40 | 0.9555 | 0.9636 | 0.9546 | 237 |
| 40-60 | 0.9417 | 0.9535 | 0.9434 | 231 |
| 60-80 | 0.9205 | 0.9367 | 0.9270 | 220 |
| > 80 | 0.9139 | 0.9282 | 0.9182 | 216 |

The train sample has the same shape. The union's median is 386 fragments, and 16.2% of molecules
have 256 or fewer. Coverage is 0.9493 for each energy's own set, 0.9546 for the union, 0.9505 at
`shared_256` (+0.0012, SE 0.0002) and 0.9528 at `shared_384` (+0.0034).

- **The per-energy sets differ, but the differences carry little intensity.** A fixed 256 matches
  the per-energy sets: about 0.001 lower up to NCE 40, better above it. At 384 it is better
  everywhere but for 1.3% of spectra.
- **Candidate channels are chemistry only.** In 10,127,438 reappearances of a fragment at another
  energy, 295 carried a different m/z list after rounding to 1e-5 Da. All had the same channel
  count, and the raw m/z differed by at most 2.3e-13 Da, a float64 last-bit difference.
- **Cost.** A shared set of 256 to 384 per molecule replaces about 12 sets of up to 256, one per
  spectrum. Construction is 5-10% of a batch today, and the intensity transformer would also run
  once per molecule. The saving is not measured.

### What a shared proposal would rank (2026-09-26)

`masster_share_proposal_across_ev` trains one CE-blind proposal per `group_id` against the
group's merged target (`_shared_group_proposal_coverage_loss`: each spectrum adds unit intensity).
In this split a group is one molecule (4,222 val groups, median 13 spectra). The loss is linear
in the target, so its optimum lies near the average of the per-energy optima, which weights
densely sampled energies more. The
[MASSter plan](../../plans/masster/masster_plan.md#shared-proposals-across-collision-energies-2026-08-29)
keeps the flag off for that reason. `share_rank.py` ranks each molecule's union by stand-ins
for what that loss would learn, and scores coverage as above.

| ranking | what it stands for | uses val targets |
| --- | --- | --- |
| `max` | best per-energy proposal probability (the `shared_N` above) | no |
| `mean` | mean per-energy probability, 0 where not kept: near the merged-target optimum | no |
| `bal` | `mean` with each occupied NCE bin weighted equally: the CE-balanced ablation the plan names | no |
| `dedup` | `mean`, but a fragment whose channel m/z values are all already in the set moves to the tail | no |
| `orc_mean` | greedy maximum coverage of the merged target | yes (oracle) |
| `orc_bal` | the same on the NCE-bin-balanced target | yes (oracle) |

Val coverage (each energy's own set 0.9402, union 0.9505):

| ranking | 128 | 256 | 384 |
| --- | ---: | ---: | ---: |
| `max` | 0.9273 | 0.9422 | 0.9471 |
| `mean` | 0.9281 | 0.9424 | 0.9472 |
| `bal` | 0.9280 | 0.9423 | 0.9473 |
| `dedup` | 0.9428 | 0.9498 | 0.9505 |
| `orc_mean`, `orc_bal` | 0.9505 | 0.9505 | 0.9505 |

Change from each energy's own set, val:

| spectra | n | own | `max` 256 | `mean` 256 | `bal` 256 | `dedup` 128 | `dedup` 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NCE ≤ 20 | 9,305 | 0.9680 | −0.0004 | −0.0002 | −0.0005 | −0.0000 | +0.0029 |
| NCE 20-30 | 5,816 | 0.9643 | −0.0009 | −0.0003 | −0.0002 | −0.0004 | +0.0052 |
| NCE 30-40 | 6,206 | 0.9555 | −0.0009 | −0.0002 | +0.0002 | −0.0003 | +0.0073 |
| NCE 40-60 | 9,916 | 0.9417 | +0.0016 | +0.0022 | +0.0028 | +0.0028 | +0.0109 |
| NCE 60-80 | 5,577 | 0.9205 | +0.0066 | +0.0065 | +0.0068 | +0.0076 | +0.0155 |
| NCE > 80 | 15,200 | 0.9139 | +0.0043 | +0.0039 | +0.0030 | +0.0047 | +0.0135 |
| molecule's lowest NCE | 4,467 | 0.9684 | −0.0007 | −0.0007 | −0.0008 | −0.0005 | +0.0008 |
| molecule's highest NCE | 4,235 | 0.9148 | +0.0022 | +0.0015 | +0.0004 | +0.0019 | +0.0111 |

- **The merged-target ranking loses no energy.** `mean` keeps 97% of `max`'s top 256 and covers
  the same (+0.0002, SE 0.0001). In every NCE bin, and at each molecule's lowest and highest
  energy, it stays within 0.0007 of each energy's own set. The rare-energy bias the plan warns
  about does not appear at these caps.
- **CE balancing buys nothing.** `bal` is within 0.0001 of `mean` overall and 0.0011 lower at
  each molecule's highest energy.
- **The merged objective solved exactly gives up nothing either.** The greedy oracle reaches the
  union's full coverage at every cap, on the merged and on the balanced target alike.
- **The union is mostly isomers.** Covering everything the union covers takes a median of 26
  fragments (p90 62, p99 109; 128 or fewer for 99.7% of molecules). Of the median 392 union
  fragments, 135 add a channel m/z that higher-ranked ones lack. This is the
  [identity-supervision finding's](masster_identity_supervision_2026_09_22.md) 2.75 connected
  fragments per formula, seen from the candidate budget.
- **Skipping same-m/z isomers is the large lever, and it needs no targets.** `dedup` at 128
  covers 0.9428, more than each energy's own 256 (+0.0027, SE 0.0003), and no NCE bin loses more
  than 0.0005. At 256 it covers 0.9498, 0.0007 short of the union. The own set's 0.0104 gap to
  the union is in the ranking, not the budget.
- **The train sample agrees.** `mean` − `max` is +0.0000 at 256. `dedup` covers 0.9504 at 128
  against 0.9493 for each energy's own set, and 0.9541 at 256 against the union's 0.9546. Full
  coverage takes a median of 26 fragments.
- **Coverage is not cosine.** A mask-order U128 cap lost about 0.02 coverage against U256 and
  still tied it on cosine
  ([fragment-marginal selection](../../plans/masster/masster_fragment_marginal_selection_plan.md#tail-read-2026-09-24-marginal-selection-fails-and-u128-is-a-free-speed-up)).
  What `dedup` buys in cosine is not measured. Its case is the budget: the same coverage from
  half the candidates.

## Fix: a smooth energy input (2026-09-26)

### Which encoding

`encoding_probe.py` compares CE encodings on the form fit's NIST20 `[M+H]+`/FT train-target CE
series.

- **Model.** Each channel's logit is `a_i + b_i · h(φ(NCE))`. `a_i` and `b_i` (4 numbers) are
  free per channel, and `h` is one shared MLP (φ → 64 → 4). That is the role the CE embedding
  and context MLP play in the models. Only φ, the fixed CE features, differs between arms.
- **Held out.** Every spectrum at seven interior NCE values (25, 27, 44, 45, 54, 88 and 110) is
  removed from the fit, so those values are never seen. The removed spectra are predicted from
  their NCE.
- **Scale.** A seeded sample of 2,000 molecules (25,119 spectra, 5,400 held out), with 2,000
  Adam steps. It ran on the CPU because another job held the GPU.
- **Reference.** Log-space interpolation between each molecule's bracketing kept energies, the
  within-molecule oracle (5,067 bracketed spectra).

| encoding φ | features | in-sample | unseen NCE values | unseen, bracketed |
| --- | ---: | ---: | ---: | ---: |
| abs-sines (the current `embed` default) | 8 | 0.9955 | 0.9480 | 0.9550 |
| RBF, 64 bumps | 64 | 0.9991 | 0.9678 | 0.9757 |
| RBF, 32 bumps | 32 | 0.9991 | 0.9627 | 0.9725 |
| RBF, 16 bumps | 16 | 0.9990 | 0.9816 | 0.9902 |
| low-frequency Fourier (periods 400/k, k = 1-4) | 8 | 0.9990 | 0.9704 | 0.9795 |
| RBF, 32 bumps, plus the `1/(NCE + 200)` scalar | 33 | 0.9991 | 0.9479 | 0.9547 |
| NCE scalar, standardized | 1 | 0.9988 | 0.9834 | 0.9947 |
| **`1/(NCE + 200)` scalar, standardized** | 1 | 0.9988 | **0.9845** | **0.9950** |
| log-space interpolation within the molecule | | | | 0.9959 |

By held-out NCE value:

| NCE | abs-sines | RBF, 32 bumps | NCE scalar | `1/(NCE + 200)` scalar |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 0.9629 | 0.9855 | 0.9887 | 0.9883 |
| 27 | 0.9639 | 0.9659 | 0.9936 | 0.9923 |
| 44 | 0.9404 | 0.9764 | 0.9804 | 0.9806 |
| 45 | 0.9921 | 0.9933 | 0.9939 | 0.9939 |
| 54 | 0.9041 | 0.9692 | 0.9796 | 0.9796 |
| 88 | 0.9342 | 0.9785 | 0.9806 | 0.9831 |
| 110 | 0.9124 | 0.8972 | 0.9685 | 0.9732 |

- **Smoothness decides how well an encoding reaches unseen energies.** Flexible local features
  let the shared MLP learn wiggles specific to each trained value, and those do not carry over.
  Fewer, wider RBF bumps do better (16 beats 32 and 64). Adding RBF features to the scalar makes
  it worse (0.9479).
- **One monotone scalar does best.** It comes within 0.001 of the within-molecule oracle on
  bracketed spectra, at a cost of 0.0003 in-sample.
- **The breakdown-curve axis edges the linear one**, mostly at high NCE: 0.9732 against 0.9685
  at NCE 110, and 0.9831 against 0.9806 at 88. Overall the difference is 0.0011.
- **abs-sines is worst both out of sample and in-sample** (0.9955 against 0.999).

### What was implemented

`ce_insert_type: qet`, in the shared `CEModel` (`src/fragnnet/model/base_model.py`) and in
`MassterContextEncoder`.

- **The encoder.** `QETEnergyEncoder` turns each raw CE reading into two features and applies a
  learned `Linear(2, ce_insert_size)`:
  - z, the standardized `1/(NCE + 200)` for NCE, so that z ≈ (NCE − mean) / std near the
    configured mean, and the standardized eV itself for ACE;
  - a missing flag for CE below 0, whose z is set to 0.

  Readings are mean-pooled per spectrum as before. There is no rounding, and `nce_max` or
  `ace_max` does not clamp it.
- **The offset of 200** is the form fit's shape parameter, measured on NCE. The ACE (eV) path
  is linear instead, following the eV measurement below (`QET_ACE_OFFSET = None`).
- **Existing behaviour is unchanged.** `embed`, `embed_clamped` and the abs-sines table stay as
  they were, so every existing config and checkpoint resolves and loads as before. The
  featurizer's docstring now records the aliasing.
- **The scalar types `id` and `lin` keep a flaw.** `transform_ce` passes a missing CE (−1)
  through unchanged, so it reads as one standard deviation below the mean. `qet` flags it
  instead.
- **Tests.** `tests/test_ce_qet_embedding.py` has 18 tests covering both paths:
  - the axis is monotone and unclamped, and unseen energies sit between their neighbours;
  - NCE keeps the 200 offset and the ACE path is linear;
  - missing CE uses the flag;
  - the NCE 76/180 and ACE 30/98 eV alias pairs are separated;
  - stepped readings pool per spectrum, and training reaches the encoder;
  - the legacy table keeps its shape.

  Run together with the existing CE, featurizer and model-construction test files, 287 tests
  pass.
- An RBF option was implemented first and replaced after the probe.

### The eV axis

`ev_axis_fit.py` fits the same one-component form to NIST20 `[M+H]+`/QTOF single-energy eV series:
the `nist20_ma_mi_inchikey` train molecules with at least 3 distinct eV (1,399 molecules, 14,964
spectra, 1-50 eV). Channels are clustered at the QTOF tolerance (20 ppm, 750 Da floor), and β is
fixed per axis:

- raw eV with one global offset, `1/(eV + n0)`;
- eV converted to NCE as the dataset does (`eV × 500 / (m/z × charge factor)`), on
  `1/(NCE + n0)`, which in eV units is an offset that scales with m/z;
- a free piecewise-linear curve over eV, the best any shared eV curve can do.

| axis | in-sample channel cosine |
| --- | ---: |
| eV, n0 = 2 / 20 / 80 / 320 | 0.9843 / 0.9923 / 0.9944 / 0.9948 |
| eV, linear | 0.99488 |
| NCE, n0 = 50 / 200 / 400 | 0.9928 / 0.9945 / 0.9947 |
| NCE, linear | 0.99488 |
| free eV curve | 0.99494 |

On one held-out interior energy per molecule (1,389 spectra), linear eV scores 0.9868. The
converted NCE on `1/(NCE + 200)` scores 0.0009 lower (paired SE 0.0001, better on 29% of spectra).
Log-space interpolation, the within-molecule oracle, scores 0.0022 higher.

- **Over 1-50 eV the axis is linear.** Every offset loses to linear, the fit rises steadily with
  the offset, and the free curve is nearly straight.
- **Converting eV to NCE and applying the NCE curvature makes it worse.** The `1/(NCE + 200)`
  shape measured on Orbitrap NCE does not carry over to QTOF eV.
- The two linear axes are the same model once each molecule's scale is absorbed; their held-out
  gap (+0.0004, SE 0.0002) is optimization noise.

So `qet` makes the ACE path linear: z = (eV − `ace_mean`) / `ace_std`, with the same missing flag.
The NCE path keeps `1/(NCE + 200)`.

### Validation arm `n1`

`config/masster/mh_ft_memorization_route_2026_09_24/n1_nce_qet.yml` is `c0` with
`ce_insert_type: qet`.

- **Config.** The resolved config differs from `c0` in that key and W&B naming only.
- **Submission.** Generate it on Fir with `submit_masster_mh_ft_memorization_route.sh n1`.
- **Smoke run.** A local 2-epoch run on a 512-spectrum subsample exits 0 (val cosine 0.064 →
  0.231).
- **Read.**
  - Against `c0` (`vcpho5qm`, tail 0.7276) on the epoch 54-63 tail. No NIST20 gain is expected,
    because every val NCE is seen in training; a drop beyond the 0.0042 seed spread would be a
    cost.
  - `nce180_check.py` on its checkpoint: unseen NCE inputs should score close to the measured
    ones.
  - Per NCE value against `c0`, split by acquisition ladder. The categorical code let the model
    treat each ladder's own values separately, and one smooth axis cannot. The
    [energy-vs-parts finding](masster_memorization_energy_vs_parts_2026_09_26.md) measured
    adjacent values of the two ladders at 0.71-0.84 cosine on the same molecule. A loss
    concentrated at the ladder-B-only values (9, 14, 27, 44, 54, 76, 88, 100) would mean the
    acquisition setup needs its own input. That needs the raw NIST `Instrument` field, which
    `spec_df` does not keep.

**Mid-run read, 2026-09-26: `n1` (`jxczo2kf`) trails `c0`.** At epoch 13 of 64, val
`cos_sim_0.01` over epochs 10-13 averages 0.7009, against 0.7127 for `c0` and 0.7137 for `c0_s2`
(`1oxhvk0n`). The gap is −0.0124, and the two `c0` seeds sit 0.001 apart there.

- `n1` is below both `c0` seeds at every epoch from 4 on, and the gap widens (−0.006 at epoch 5,
  −0.014 at epoch 13).
- Across the 17 finished arms of this group, the epoch 10-13 gap to the `c0` pair predicts the
  epoch 54-63 gap at r = 0.84. The fit projects `n1` at −0.011 (residual SD 0.005). `gm5` had the
  same early gap and finished at −0.005.
- Val `candidate_wrecall` matches `c0` within 0.001 at every epoch, so the loss is in intensity
  allocation, not the candidate set.
- Epoch time equals `c0`'s (3.1 min), so the wall-clock comparison gives the same answer.

**Mechanism, as far as the code goes.** In MASSter the CE embedding enters the context projector
(Linear → SiLU → Linear, `MassterContextEncoder`), next to the adduct and instrument embeddings.
`n1`'s `Linear(2, 32)` output is affine in z, and its missing flag is constant on this split,
which has no missing NCE. So the projector's first layer can absorb it: `n1` is a standardized
scalar on the `1/(NCE + 200)` axis entering an MLP. `embed` instead gives each integer NCE its own
32-wide code, which the projector can use per energy. Every val NCE also occurs in training, so
that per-energy code can only help on NIST20 val.

**Mid-run read 2, W&B at 03:45 UTC on 2026-09-27: `n1` at epoch 33; `n2` (`zov75vk3`) and `n3`
(`o9pz0j9j`) at epoch 9.** Train loss is train-mode primary loss, val loss eval-mode, both
against `c0` at the same window. The loss settings are identical, so these compare.

| arm | window | cosine vs `c0` | train loss vs `c0` | val loss vs `c0` |
| --- | --- | ---: | ---: | ---: |
| `n1` | 24-33 | −0.0137 | −0.002 | +0.055 |
| `n1` | 6-9 | −0.0077 | −0.004 | +0.016 |
| `n2` | 6-9 | −0.0050 | +0.002 | +0.003 |
| `n3` | 6-9 | −0.0069 | −0.007 | +0.013 |

- **`n1` fails gate 1.** It is below `c0` at 32 of 34 epochs and below `c0_s2` at all 34. The
  gap is steady at −0.009 to −0.015 from epoch 20.
  - The early-gap line over the 21 finished arms of this recipe has r = 0.98 at this window.
    It projects the tail at −0.012 (residual SD 0.0014), three times the seed spread.
  - Coverage matches `c0` within 0.0002, and epoch time is equal.
- **The cost is on val only.** `n1` fits train as well as `c0` at every window (train loss −0.002
  to −0.008), while val loss is worse by +0.019 to +0.055 and growing. The per-energy code is not
  capacity that train needs. It is information that transfers to new molecules. The ladder reading
  predicts exactly this.
  - On a train molecule, the model can recover the acquisition ladder from the molecule itself:
    only 535 of 22,411 molecules have values from both ladders. On a val molecule, only the NCE
    value names the ladder, and a smooth axis blurs it.
  - This is still inference from the loss pattern. The per-NCE-by-ladder read on `n1`'s
    checkpoint is the test.
- **`n2` and `n3` are too early to resolve.** At epochs 6-9 the early-gap line has r = 0.75 and
  a residual SD of 0.005.
  - The losses lean one way: `n3` follows `n1` (train lower, val +0.013), as an axis-curvature-only
    difference should. `n2` is near `c0` on both.
  - Read both at about epoch 30, where the line reaches r ≈ 0.98.
- **Hold.** `qet` should not go into new configs (Spectraverse cv0-cv9, Enveda `i0`) until
  `n2` and the ladder read are in. A ladder-blind smooth input costs about 0.012 on the
  seen-energy NIST20 split. Spectraverse's continuous derived NCE is the case `qet` was built for,
  so the trade there is unmeasured, not negative.

**Follow-up arms `n2` and `n3` (built 2026-09-26; submitted 2026-09-26, read above).** Both are in the same config
directory and submitter (stage `nce`).

| arm | config | change | controls |
| --- | --- | --- | --- |
| `n2` | `n2_nce_qet_mlp.yml` | `n1` with `ce_insert_type: qet_mlp` | `n1`, `c0` |
| `n3` | `n3_nce_id.yml` | `c0` with `ce_insert_type: id` | `c0`, `n1` |

- **`qet_mlp`** is a new option: the same (z, missing flag) input through
  Linear(2, 32) → SiLU → Linear(32, 32); the hidden width is `ce_insert_size`. `qet` is
  unchanged, and its checkpoints still load, because its layer keeps the name `linear`.
- **`id`** feeds (NCE − `nce_mean`) / `nce_std` into the projector directly: a CE feature one wide.
  It differs from `n1` by the axis curvature only, up to the linear layer the projector absorbs.
- **Read.** `n3` near `n1` means the cost comes from giving up the per-energy code, whatever the
  axis. `n3` clearly apart from `n1` means the axis shape matters. `n2` near `c0` means a smooth
  axis needs depth on energy alone. `n2` near `n1` means depth on z does not help. Both keep
  `n1`'s unseen-NCE check, since both extrapolate smoothly.
- **Configs resolve** to their base plus `ce_insert_type` and W&B naming only. Local 2-epoch
  smokes on a 512-spectrum subsample exit 0 (val cosine `n2` 0.070 → 0.226, `n3` 0.068 → 0.222).

**Arm `n4`: the per-NCE code without aliasing (built 2026-09-26, not submitted).**
`n4_nce_fourier.yml` is `c0` with `int_embedder: fourier`; `ce_insert_type` stays `embed`.
`n1`-`n3` suggest the per-NCE code carries something that transfers, so this arm keeps the code
and replaces the table that aliases.

- **The table.** `fourier` is sin and cos at the same powers of 1/2 as abs-sines, down to period
  512. No two integers in 0-200 or 0-300 share a row: the nearest two are 1.41 apart, where 76 and
  180 are 5.5e-6 apart under abs-sines. At max 300, abs-sines has 44 colliding pairs among the
  integers.
- **Scope in MASSter.** `int_embedder` reaches only the CE table. The table feeds the existing
  `Linear(num_dim, 32)`, whose input goes from 8 to 20 columns. In FraGNNet the same key also
  embeds formula counts, depth, RDBE and the other integer features, so there it is not CE-only.
- **Expected effect here: none.** On this split, 76/180 is the only colliding pair that occurs
  (2.2% of train spectra, 2.1% of val). No molecule in train or val has both, and the NCE-180
  val spectra lost nothing measurable ([table above](#the-nce-input-is-categorical)).
  `n4` prices the table swap before `fourier` becomes the default per-NCE table for
  datasets with many distinct energies. It does not fix unseen energies: its high-frequency
  columns stay arbitrary there.
- **Read.** The tail against `c0` (`vcpho5qm`, 0.7276): inside the 0.0042 seed spread passes, and
  a drop beyond it rules the table out. On the best checkpoint, compare val cosine at NCE 76 and
  at 180 against `c0`.
- **Checks.**
  - Through `load_config`, `n4` differs from `c0` by `int_embedder` and W&B naming only.
  - A local 2-epoch smoke on a 512-spectrum subsample (batch 64) exits 0, with val cosine
    0.071 → 0.229. Its checkpoint holds the 201 × 20 table and `Linear(20, 32)`. The projected
    embeddings of 76 and 180 differ by 99% of their norm.
  - The submitter's config checks pass for `n4`; locally it stops at the missing `pyteomics`
    wheel before generating anything.
- **Generate** on Fir with `submit_masster_mh_ft_memorization_route.sh n4` (`fir:h100-3g.40gb`),
  after committing the config and the submitter change.

**NCE scale check.** Every config in this directory uses the template's `nce_max: 200`,
`nce_mean: 60`, `nce_std: 40`. The NIST20 `[M+H]+`/FT split matches them:

| set | spectra | NCE mean | NCE std | range | distinct values |
| --- | ---: | ---: | ---: | --- | ---: |
| train | 157,678 | 62.2 | 42.6 | 2-195 | 42 |
| val | 52,020 | 62.1 | 42.5 | 2-195 | 42, all in train |

- No NCE exceeds 200, so `embed` clamps nothing, and no spectrum is stepped.
- On `qet`'s axis the train NCE has z mean −0.11 and std 0.99. For `id` the scalar has mean 0.06
  and std 1.07. Both are already standardized.
- Over all NIST20 FT spectra, NCE runs up to 195 (mean 58.4, std 39.4). QTOF eV runs up to 70
  (mean 22.3, std 13.1), inside the template ACE scale (20, 15, max 100).

## What this means for using QET in MASSter

- **Not as an accuracy fix for the CE law.** On the energies it was trained on, `u2` already
  produces QET-consistent series. Enforcing the law directly is worth about +0.002 on val. The val
  error is which channels a new molecule opens, and its onset, at every energy at once. The
  form-fit finding reached the same conclusion for survival alone.
- **As the energy input, which is the cheapest change and is now implemented.** QET says energy
  enters through one monotone scalar, and the form fit measured it as `β ∝ 1/(NCE + 200)`. The
  models instead fed a binary-like code in which neighbouring energies are unrelated and unseen
  ones fail. `ce_insert_type: qet` feeds that axis, plus a missing-CE flag
  ([Fix](#fix-a-smooth-energy-input-2026-09-26)).
  - On NIST20 val the expected gain is small: every val NCE is also in training, and NCE
    165-195 hold 0.7% of val spectra. Arm `n1` checks that the switch costs nothing.
  - The payoff is transfer to other NCE sets: Enveda-180, Spectraverse, MassSpecGym and QTOF eV.
    New configs for those should use `qet`.
- **As a compute restructuring, it is feasible.** A CE-blind candidate set of 256 loses nothing
  on average, and 384 gains. The encoder, candidate construction and intensity transformer could
  run once per molecule, with one softmax per measured energy.
  - A proposal trained on the merged target should rank like the per-energy average, and that
    ranking serves every energy as well as the best-score one
    ([section](#what-a-shared-proposal-would-rank-2026-09-26)).
  - Skipping same-m/z isomers gives from 128 candidates the coverage that 256 give today.
  - `share_proposal_across_ev` has never been read. Three August unmerged runs crashed by epoch
    5, and merged runs have about one spectrum per group, so sharing does nothing there. Arms
    `sp0` (`c0` + whole-molecule CE-context batches, which sharing requires) and `sp1` (`sp0` +
    sharing) are built on the memorization-route lineage; both pass a 2-epoch local smoke run.
- **Training dynamics are the open question.** Sharing `(a_i, e_i)` across a molecule's energies
  could change what the network memorizes, and a projection of a trained model cannot show that.
  The CE-dimension generalization gap is small (projection +0.0019 on val, −0.0009 on train), so
  a large effect is not expected.
- **For stepped and multi-instrument data** (Enveda-180, Spectraverse), the mixture form
  `Σ_k w_k softmax(a − e·β(CE_k))` and per-instrument β curves are untested. None of this
  measures them.

## Limits

- **One checkpoint.** `u2` is the EMA lineage. The NCE featurizer is shared by every MASSter
  config, but how much a given run relies on it is not measured.
- **Projection is post hoc.** It shows how far the trained model's outputs are from the law, not
  what training under the law would learn.
- **The curves are held fixed from train targets** (M1g, M2g). M1n frees β per NCE value. No
  variant refits curve shapes on the predictions beyond that.
- **The rankings come from the trained CE-conditioned proposals.** `mean` stands in for what a
  CE-blind proposal trained on the merged target would learn. A trained one may rank
  differently, and here it ranks the union of about 12 per-energy selections. One CE-blind pass
  draws from a single enumeration, so `dedup` there would need enumeration past today's cap.
  Only a run shows either.
- **The ladder reading is a hypothesis.** The acquisition setup is not in `spec_df`, and the
  ladder is only a proxy for it.
- **The encoding probe is a proxy for the models' CE pathway.** Its per-channel coefficients are
  free rather than predicted from structure. It used 2,000 molecules and 2,000 CPU steps, and it
  holds out whole NCE values, not new molecules. Its free coefficients absorb any offset a
  molecule's acquisition ladder adds, so it cannot see what a single axis loses there. Only a
  training run (`n1`) shows what `qet` does inside MASSter.
- **The eV axis was measured on 1-50 eV QTOF series only.**
  - Nothing above 50 eV was tested. Dual-CE configs give Orbitrap spectra ACE values derived
    from NCE and m/z, reaching several hundred eV. On the linear axis z grows without bound:
    300 eV at the template `ace_mean: 20` and `ace_std: 15` gives z ≈ 19. A dual-CE config that
    adopts `qet` should set `ace_mean` and `ace_std` for its data.
  - Whether eV should be normalized by m/z cannot be tested with per-molecule fits: within a
    molecule the conversion is a constant factor, which the per-channel parameters absorb.
- **NIST20 `[M+H]+` only** (FT for NCE, QTOF for eV), single-energy spectra.

## Files

`output/qet_projection_2026_09_26/`:

- `curves.pkl`.
- `u2_{val,train}.pkl` (collected), `u2_{val,train}_proj.pkl`, `u2_{val,train}_cov.pkl`.
- `u2_val_mismatch.pkl`, `u2_val_nce_sweep.pkl`.
- A `_log.txt` per step, including `nce180_check_log.txt`.
- `u2_val_proj_v1.pkl`, the first val projection before M1n was added; its other columns agree to
  1.3e-4 per spectrum.
- `collect_train_oom_log.txt`, the killed first train run.
- `encoding_probe.py`, `encoding_probe_log.txt`, `encoding_probe.pkl` and
  `encoding_probe_baselines.pkl`: the CE-encoding comparison (run with `PROBE_DEVICE=cpu
  PROBE_MOLS=2000 PROBE_STEPS=2000`).
- `smoke_n1/`, `smoke_n2/`, `smoke_n3/`: the arm smoke runs' logs.
- `share_rank.py`, `u2_{val,train}_share_rank.pkl` and `share_rank_{val,train}_log.txt`: the
  shared-proposal rankings.
- `ev_axis_fit.py`, `ev_channels.pkl`, `ev_axis_fit.pkl`, `ev_axis_heldout.pkl` and
  `ev_axis_fit_log.txt`: the eV-axis fit (`ev_axis_fit_partial_log.txt` is an interrupted first
  run).
