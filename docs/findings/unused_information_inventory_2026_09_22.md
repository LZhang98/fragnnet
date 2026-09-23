# What information do we hold and never use?

> **Status:** FINDING — OPEN, the largest item is actionable &nbsp;·&nbsp; **Updated:** 2026-09-22
>
> The biggest unused input in this repository is not a feature. **NIST ships an expert formula
> assignment for 95.8% of peaks carrying 99.4% of intensity, on 68.4% of our spectra, and it
> never reaches the training loss** — only the GrAFF baseline reads `ann_df`. Measured against
> our DAG, ~8% of annotated intensity is scored as explained by a formula whose heavy-atom
> skeleton NIST disagrees with and the DAG cannot build.

Companion to [feature headroom](feature_headroom_2026_09_22.md), which closed the question for
descriptor features. This asks the complementary question: not "what could we compute", but
**"what do we already have and throw away"**.

## Inventory, measured

| source | size | used today |
|---|---|---|
| **NIST peak annotations** (`ann_df.pkl`, `ann_products` / `ann_losses` / `ann_isotopes`) | 397,057 spectra = **68.4%** of the `ma_mi` subset; **95.8%** of peaks; **99.4%** of intensity | `src/fragnnet/graff/` (a baseline) and isotope-detection tuning. **Never in FraGNNet or MASShatter training** |
| **ClassyFire taxonomy** (`data/classyfire/nist20_mol_classyfire.pkl`) | **100%** of 35,025 molecules: 20 superclasses, 341 classes, 795 subclasses, 1,497 direct parents, plus a **2,301-term substituent inventory**, mean 21.9 terms/molecule | never a model input. A [pretraining plan](../plans/fragnnet/classyfire_atom_pretraining_plan.md) exists, PROPOSED / NOT STARTED |
| **DAG route structure** (`in_degree`, `out_degree` fragment features) | implemented; 87.7% of nodes are multi-parent, so `in_degree` has real variance | present only in `config/template.yml` (commented) and one debug config. **Never in a production run** |
| **Replicate spectra** (`group_id` + identical mol/adduct/instrument/CE) | 26,205 cells with ≥2 replicates, covering **12.1%** of spectra | merging only; no per-peak reliability weight |
| `res` (per-spectrum resolution) | 13 codes, varies *within* instrument | unused; the loss tolerance is per-instrument |
| `col_gas`, `ce_type`, `*_extra_*` | He is 0.5%, no stepped spectra, 100% null | measurably dead — see [feature headroom](feature_headroom_2026_09_22.md) |

## The headline: our coverage includes credit for the wrong molecule

`scripts/debug/debug_annotation_vs_dag_identity.py`. Peak coverage is normally measured by m/z
matching, which cannot separate a correct assignment from a coincidental one. NIST's annotation
can. For 400 molecules / 192,463 annotated monoisotopic `[M+H]+` peaks, at 10 ppm:

| outcome | peaks | intensity |
|---|---:|---:|
| no DAG formula within tolerance | 0.335 | 0.120 |
| DAG matches and **NIST's formula is in the DAG support** | 0.567 | 0.794 |
| mismatch, but the heavy skeleton **is** reachable (hydrogen only) | 0.008 | 0.006 |
| **mismatch, heavy skeleton NOT reachable** | **0.091** | **0.081** |

The last row is the finding: those peaks score as *explained* under any m/z-matched coverage or
oracle metric, while the composition doing the explaining is one NIST says is not there, and the
composition NIST does name cannot be built by the DAG at all. The loss sees m/z and intensity
only, so on that slice the model is actively rewarded for the wrong assignment. This is a
shortcut the training signal cannot distinguish from the truth — the concrete form of the
concern in [feature headroom](feature_headroom_2026_09_22.md#second-pass-shortcut-learning-2026-09-22).

**Checks this survived.** The ion-formula convention is verified, not assumed: the neutral-atom
mass of `ann_products` matches NIST's own `ann_exact_mzs` to within one electron mass on
**100.00%** of a 121,703-peak sample, which licenses the `neutral = ion − H` conversion. The
hydrogen-only row rules out an h-window or convention artifact: it is 0.6% of intensity against
8.1% for real skeleton mismatches.

**Limits.** NIST annotations are assignments, not ground truth; 0.72% of annotated products
carry a chemically implausible RDBE. And the estimate is molecule-set sensitive — a 40-molecule
smoke run gives 5.5% intensity rather than 8.1% — so **re-measure at corpus scale before acting
on the size of the effect**. The direction is not in doubt; the magnitude is.

## What each source is good for, ranked

1. **Annotations as auxiliary identity supervision.** The current loss cannot distinguish two
   candidates at one m/z; an existing measurement puts real ambiguity at mean 1.30 candidate m/z
   clusters per peak with >1 on 28.8% of peaks. NIST's label resolves exactly those. Note the
   standing guidance to avoid hard BCE targets built from m/z-only many-to-one matching and
   prefer noisy-OR / responsibility — that guidance was about labels *derived from m/z matching*,
   which are circular. NIST's are external, so hard targets are more defensible here, but the
   safer form still applies.
2. **Annotations as a labelled list of missing chemistry.** The uncovered and skeleton-mismatch
   rows are a machine-readable residual: every NIST formula the DAG cannot build, with the
   molecule it came from. SMARTS rule mining currently finds these by hand. The 40-molecule smoke
   run surfaced an all-fluorine cluster (`C6F1H10N1O1`, `C7F1H9N1O1`, …), which is the shape a
   mined rule starts from.
3. **Annotations as isotope labels.** `ann_isotopes` marks isotope peaks directly, and isotopes
   have never been exercised in training (`num_isotopes` is 1 everywhere, every DAG `isoFalse`).
4. **ClassyFire as conditioning.** Per-class cosine gaps of +0.10 are a standing priority item
   and the labels are already computed for every molecule. The taxonomy is derivable from
   structure in principle, so by the [derivability argument](feature_headroom_2026_09_22.md) it
   adds inductive bias rather than information — but the substituent inventory is a curated
   2,301-term vocabulary, far beyond the handful of SMARTS flags that probe tested.
5. **`in_degree` / route multiplicity.** A fragment reachable by many routes is a different
   object from one reachable by a single route. This is a DAG-structural property, **not** a
   molecular descriptor, so the derivability probe does not cover it. Implemented, free, never
   run. Informativeness unmeasured.
6. **Replicates** reach only 12.1% of spectra; a reliability-weighted loss would leave most of
   the corpus untouched.

## Next concrete check

Re-run `debug_annotation_vs_dag_identity.py` at corpus scale, stratified by collision energy and
adduct, to pin the mismatch magnitude and see whether it concentrates where the
[CE decomposition](glacier_gap_ce_decomposition_2026_09_21.md) says our loss already is. Nothing
here has been trained.
