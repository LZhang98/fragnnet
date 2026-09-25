# Does local bond chemistry know what MASSter misses on new molecules?

> **Status:** FINDING · **Updated:** 2026-09-24
> Phase 1.4b of the [intensity-displacement plan](../../plans/masster/masster_intensity_displacement_plan.md),
> the offline gate for arm 2A (an identity-blind local lability prior). Checkpoint f1
> (`6h67zps9`), unmerged `[M+H]+`/FT, eval mode; no training of MASSter.
>
> - **The local learner carries real information the model lacks, but not much.** Stacked onto
>   the model's fragment scores, it adds +0.0022 ± 0.0003 group cosine on val and
>   +0.0025 ± 0.0003 on test over a fitted temperature. Beyond the strongest control without a
>   learner (a temperature and a precursor offset per NCE bin), it adds +0.0017 on val and
>   +0.0028 on test.
> - **It helps where transfer fails.** Beyond that control, the gain is +0.0032 (val) and
>   +0.0037 (test) on novel molecules (nearest-train Tanimoto < 0.5), against +0.0012 and
>   +0.0024 on familiar ones. By NCE it is +0.003 to +0.005 at NCE ≤ 40 and +0.0006 above 60.
> - **It is not precursor survival, and not the bond-type × CE trend.** The learner applied
>   only to the precursor adds 0.0000. A learner that sees only bond type, NCE and candidate
>   count adds +0.0006 / +0.0004. The model's own bond-share aggregate adds 0.0000.
> - **Alone, the local learner orders a novel molecule's bonds worse than MASSter.** On
>   cross-type bond pairs, mass-weighted pairwise accuracy is 0.875 against the model's 0.904
>   (val, novel). It complements the model; it does not beat it.
>
> Gate: the pre-registered PASS needs +0.005 over `model_T` on both splits and FAIL is below
> +0.002. At +0.0022 / +0.0025 the result sits between them. 2A is not built on this
> evidence; see [Consequences](#consequences-for-the-plan).

## Question

The [bond-type check](masster_intensity_displacement_phase0_2026_09_24.md#addendum-bond-type-check-iceberg-style-2026-09-24)
put 59% of MASSter's val−train bond-level gap between bond types within a molecule, 26%
within a type and 16% at the precursor. Arm 2A would add a prior built only from local
structure around each bond, so it cannot memorize molecules. The prior is useful only if local
structure predicts, on unseen molecules, something about fragment ordering that the model gets
wrong. This test measures that offline, in the form 2A would take.

## Method

- **Records.** [`scripts/eval/bond_type_ce_allocation.py`](../../../scripts/eval/bond_type_ce_allocation.py)
  gained two options. `--bond_records_fp` writes, per spectrum, every candidate bond's model
  mass and target mass. `--fragment_level` adds each retained fragment's cut bonds and each
  valid channel's fragment, m/z group and model mass, plus the target mass per group. The
  retained fragment set depends on NCE (up to 256 per spectrum, fewer at high NCE), so these
  are stored per spectrum.
  - Train: 80 shuffled batches, 40,960 spectra of 12,055 molecules.
  - Val: 51,999 spectra, 4,221 molecules.
  - Test: 52,621 spectra, 4,223 molecules.
- **Learners.** [`scripts/eval/bond_local_learner.py`](../../../scripts/eval/bond_local_learner.py),
  tests in [`tests/test_bond_local_learner.py`](../../../tests/test_bond_local_learner.py).
  - Fitted on the train molecules, with 10% held out for early stopping.
  - Target: the bond's share of target mass, with shared-m/z mass split equally (model-free).
  - Inputs, 75 per bond:
    - bond order, aromaticity, conjugation, ring membership and ring size;
    - both endpoint atoms (element, degree, H count, charge, hybridization, ring, heteroatom
      neighbours, attached C=O, aromatic neighbour);
    - element and aromatic-atom counts at one and two bonds;
    - for a non-ring cut, the formula and size of the lighter released piece;
    - molecule heavy-atom count and mass, NCE and the candidate count.
  - Never the molecule's identity or any model output.
  - `gbdt`: HistGradientBoosting, Poisson loss (1,000 iterations).
  - `mlp`: 3×256 MLP with the model's listwise CE.
  - `gbdt_type`: a GBDT on bond type, NCE and candidate count only.
- **Peak-level stack (the gate).**
  - Each fragment is re-scored from its model log-mass `log w_f` and from the mean, over its
    cut bonds, of the model's bond log-share (`B_model`) and of each learner's score (`B_*`).
    The intact precursor uses its own row.
  - The scores are renormalized per spectrum and split over ion/H channels as the model does.
    Channels are summed into m/z groups (1e-5 rounding) and scored against target peaks matched
    within 10 ppm.
  - Weights are fitted by group-level cross-entropy on 40% of the other split's molecules and
    applied to every spectrum of this split. Val and test share no molecules.
  - The metric is group cosine, with unmatched target peaks kept in the target norm. Its level
    is not the production cosine; only differences are read. Standard errors are clustered by
    molecule.
- **Controls.**
  - `model_T`: one temperature. It includes the free `P^0.75`; the fitted value is 0.733.
  - `stack_logp`: the model's own bond-share aggregate.
  - `ctrl_nce`: a temperature and a precursor offset per NCE bin (≤20, 20-40, 40-60, >60).
    Added after the first read.
- **Why not bond level.** Isomeric fragments share an m/z, so which bond a target peak belongs
  to is a convention, not an observation. On train, the model's bond-level TV is 0.25 under the
  equal split and 0.07 under the responsibility split. The gate was therefore moved from type TV
  at bond level to cosine at peak level, after this train check and before any val/test number
  was seen. Bond-level tables are descriptive only.

## Result

### The learner adds about +0.002 to +0.003 beyond temperature

Group cosine for all spectra, with gains over `model_T` (± molecule-clustered SE):

| predictor | val | test | val − `model_T` | test − `model_T` |
|---|---|---|---|---|
| `model_raw` | 0.7620 | 0.7655 | | |
| `model_T` (one temperature) | 0.7667 | 0.7697 | | |
| `stack_logp` (model's bond aggregate) | 0.7667 | 0.7696 | −0.0000 ± 0.0000 | −0.0000 ± 0.0000 |
| `stack_gbdt_type` (bond type × NCE) | 0.7673 | 0.7701 | +0.0006 ± 0.0001 | +0.0004 ± 0.0001 |
| `stack_mlp` | 0.7683 | 0.7715 | +0.0016 ± 0.0003 | +0.0018 ± 0.0003 |
| `stack_gbdt` | 0.7690 | 0.7721 | +0.0023 ± 0.0003 | +0.0025 ± 0.0003 |
| `stack_all` (gbdt + mlp; gate predictor) | 0.7689 | 0.7721 | **+0.0022 ± 0.0003** | **+0.0025 ± 0.0003** |
| `ctrl_nce` (T and precursor offset per NCE bin) | 0.7677 | 0.7705 | +0.0010 ± 0.0001 | +0.0008 ± 0.0001 |
| `ctrl_nce` + gbdt on cut fragments + gbdt on precursor | 0.7695 | 0.7732 | +0.0028 ± 0.0004 | +0.0036 ± 0.0003 |

The GBDT carries it; the MLP adds nothing on top. The fitted weights are stable across the
cross-fit. Fitted on val: `a = 0.750` on `log w_f`, `c = −0.131` on `B_model`,
`b = 0.262` on `B_gbdt`. Fitted on test: 0.743, −0.124, 0.268. The prior enters as a partial
correction of the model's bond shares, `≈ 0.26 (g_b − 0.5 log p_b)`.

### The gain sits on novel molecules and below NCE 60

| stratum | spectra (val / test) | `model_T` cos | `stack_all` − `model_T` (val / test) | `ctrl_nce`+gbdt − `ctrl_nce` (val / test) |
|---|---|---|---|---|
| familiar (Tanimoto ≥ 0.5) | 37,134 / 37,252 | 0.809 / 0.810 | +0.0019 / +0.0023 | +0.0012 / +0.0024 |
| novel (< 0.5) | 14,865 / 15,369 | 0.662 / 0.673 | +0.0030 / +0.0029 | +0.0032 / +0.0037 |
| NCE ≤ 20 | 9,294 / 9,622 | 0.845 / 0.849 | +0.0048 / +0.0037 | +0.0034 / +0.0054 |
| NCE 20-40 | 12,012 / 11,946 | 0.783 / 0.791 | +0.0041 / +0.0051 | +0.0031 / +0.0051 |
| NCE 40-60 | 9,916 / 9,995 | 0.758 / 0.763 | +0.0014 / +0.0023 | +0.0010 / +0.0021 |
| NCE > 60 | 20,777 / 21,058 | 0.726 / 0.725 | +0.0004 / +0.0005 | +0.0006 / +0.0006 |

SEs are 0.0003-0.0009 throughout. Above NCE 60 (40% of spectra and the lowest cosine), local
bond chemistry adds almost nothing.

### What the gain is not

- **Not precursor survival.** The GBDT score applied only to the precursor fragment adds
  +0.0000 (val) and −0.0000 (test). Applied only to cut fragments with one global temperature,
  it adds +0.0014 / +0.0010. With the per-NCE precursor offsets of `ctrl_nce` it adds
  +0.0027 / +0.0034. Its content is the ordering among cut fragments, but it needs the precursor
  re-anchored per NCE, and one shared coefficient only does that partly.
- **Not the bond-type × CE trend.** `stack_gbdt_type` adds +0.0006 / +0.0004. This agrees with
  the bond-type check, where aggregate type shares were already right.
- **Not a re-weighting of the model's own bond shares.** `stack_logp` adds 0.0000.

### Alone, the learner is worse than the model

Bond level, with the equal-split target convention (descriptive only). Mass-weighted pairwise
ordering accuracy:

| | cross-type pairs, novel (val / test) | cross-type pairs, familiar (val / test) | precursor vs bond, novel (val / test) |
|---|---|---|---|
| `model_T` | 0.904 / 0.907 | 0.920 / 0.922 | 0.963 / 0.962 |
| `gbdt` alone | 0.875 / 0.872 | 0.898 / 0.898 | 0.948 / 0.950 |
| `mlp` alone | 0.875 / 0.870 | | 0.944 / 0.946 |
| `gbdt_type` alone | 0.785 / 0.782 | | 0.933 / 0.936 |
| `stack_all` | 0.910 / 0.910 | 0.927 / 0.929 | 0.964 / 0.964 |

Even on molecules it has not seen, MASSter orders bonds of different types better than a
local rule does. The rule adds a small complementary piece, not a better ordering.

## Reading

- **The premise of 2A holds only weakly.** Local structure predicts something about fragment
  ordering on new molecules that the model gets wrong. The effect is concentrated where Phase 0
  found the structured displacement: novel molecules at NCE ≤ 40. It is +0.002 to +0.004
  cosine, about a third of what weight EMA gives (+0.009 / +0.010 on the two lineages) and an
  eighth of a two-run ensemble (+0.020).
- **The novel-molecule deficit is mostly not local-chemistry ordering.** Novel molecules sit
  0.15 cosine below familiar ones under `model_T` (0.662 against 0.809 on val). The local
  learner closes +0.003 of that. The rest is something a bond-local, identity-blind rule cannot
  see from these inputs.
- **High CE is not a bond-lability problem.** Above NCE 60 the learner adds +0.0006. The error
  there, the largest in cosine, is not which bond breaks first. That is consistent with high-CE
  spectra being products of consecutive dissociations.

## Consequences for the plan

- **Gate G1 is not passed.** 2A is not submitted on this evidence.
- **One offline step could still move the verdict, and it is cheap.** The learner here is
  trained on a bond-share convention and uses element counts, not environment hashes. The step
  is to train the same inputs directly on the peak-level objective (fragment scores against m/z
  groups on train) and add hashed radius-2 environments.
  - Build 2A if that reaches +0.005 beyond `ctrl_nce` on both splits.
  - Close 2A if it stays near +0.003.
  - The record files already exist, so the step needs no new inference.
- **The NCE-binned temperature with a precursor offset is a free inference-time +0.0010 /
  +0.0008 over one global temperature.** It belongs with conditional sharpness (plan 1.5), and
  alone it is below the +0.005 G1-inference bar.

## Limits

- One checkpoint (f1) and one target (its filtered NIST20 target).
- Group cosine is not the production metric. The gains are differences on the same spectra and
  groups.
- The stack is linear with one or two learner weights. A prior trained inside the model could
  extract more (jointly fitted) or less (the model may absorb part of it).
- Stack weights are fitted on 40% of the other split's molecules.
- `ctrl_nce` and the precursor/bond decomposition were added after the first read. They are
  stricter controls, not a relaxed gate.
- The inputs include molecule size, a deliberate deviation from the plan's "no molecule-level
  feature". Size is not identity.
- The train slice covers 12,055 molecules but only 40,960 of the ~158k train spectra.

## Files

Local, Git-ignored, under `output/cross_eval_mh_ft_clean_2026_09_23/analysis/displacement/bond_local/`:

- `bonds_train.pkl`, and `bonds_valtest.pkl` (2.1 GB, with fragment-level arrays);
- `types_*.pkl`;
- `summary.json`;
- `peak_metrics_{val,test}.pkl` and `bond_metrics_{val,test}_{t_eq,t_resp}.pkl`;
- `run1/`, the first read before the controls were added; its shared predictors are identical.

Runtime: the probe takes about 22 min on the local RTX 2070 Super (train 4.5 min, val and test
about 8 min each). The learner step takes 11-16 min on CPU, with a 16 GB peak.
