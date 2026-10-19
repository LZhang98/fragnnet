# AIMe / DeepMS2Reasoner (Cornell + BTI, bioRxiv 2026-08-06)

> **Status:** NOTE &nbsp;·&nbsp; **Updated:** 2026-09-28
>
> Literature notes on "Charting the small-molecule universe from mass spectra with neuro-symbolic
> AI". Read the **published-metric caveat** below before quoting any of its cosines against ours.

- DOI: [10.64898/2026.08.05.743095](https://www.biorxiv.org/content/10.64898/2026.08.05.743095v1)
- Project page: <https://www.cs.cornell.edu/gomes/udiscoverit/kosmos/> · code promised at
  <https://github.com/gomes-lab/aime> (not public as of 2026-09-20)
- Local copies (gitignored): `data/paper/AIMe_DeepMS2Reasoner_biorxiv_2026.08.05.743095.pdf`
  and the extracted `.txt` beside it. Line numbers below refer to the `.txt`.
- Acikalin, Feng, Ferber, Gouveia et al.; corresponding Gomes (CS Cornell) and Schroeder (BTI).
- Companion note: [`glacier_object_detection.md`](glacier_object_detection.md). Same shape of
  protocol problem in a different place — AIMe cleans the **target spectra**, GLACIER scores at a
  **bin width 100x coarser than the instrument**. Neither set of numbers can sit beside ours
  without redoing the protocol.

## Published-metric caveat — READ FIRST

**The headline cosines are measured against cleaned target spectra, not raw experimental ones,
on two of the three benchmarks.** From Methods (line 870):

> "Peaks were merged within 20 ppm mass tolerance and **filtered to retain signals explainable
> by chemically feasible molecular subformulas within 5 ppm; spectra with no explainable peaks
> or large unexplained peaks were discarded.**"

That removes exactly the peaks a formula- or DAG-constrained model cannot reach, and discards
whole spectra dominated by them. The paper reports **no retention statistics** and **no threshold**
for what counts as a "large unexplained peak".

**How much is it worth? Probably +0.01 to +0.05, not the whole gap.** We run the same filter
(`filter_subformula_peaks`, off by default) and have measured its retention on
`nist20_ma_mi_inchikey`, FT `[M+H]+`, post-fix: **97.2% of intensity kept, 88.0% of peaks kept,
99.2% of base peaks kept, 0% of spectra emptied**. Cosine is intensity-weighted, so ~3%
unexplained intensity caps an otherwise-perfect prediction near 0.985. Their 5 ppm on
20-ppm-merged peaks is stricter than our instrument tolerance (~10 ppm FT), so treat ours as a
lower bound on how much they removed. Conclusion: the filter is a real, undisclosed thumb on the
scale, but it does not by itself explain 0.83 vs our 0.786 — the rest is real modelling and/or
the other protocol differences (80% training data, `[M+H]+`/`[M-H]-` only, no IT/CID).

The **weaker part of their protocol is the spectrum discard**, not the peak filter. Dropping peaks
changes the metric, which is replicable and correctable; dropping spectra with "large unexplained
peaks" is test-set selection on a criterion correlated with model failure, and with no stated
threshold it cannot be reproduced or bounded. By our table, base-peak-unexplained is ~1% of
spectra, so the likely magnitude is small — but that is our estimate, not their disclosure.

**We are not clean here either.** `clean_spectra` runs at dataset load
([`base_dataset.py:430`](../../src/fragnnet/dataset/base_dataset.py)) over the whole `spec_df`,
before the split, so every `..._subformula_filter.yml` config filters our val and test targets
too. See [subformula-filter A/B verdict](../findings/subformula_filter_ft_qtof.md#fragnnet-ab-verdict-and-live-hazard-from-local-memory-wb-numbers-not-re-queried-2026-10-06): "Neither Δcosine is a matched eval… each arm
is scored against its own cleaned target", and the uniform +0.026 `val_candidate_wrecall` was a
denominator artifact. Any future claim of ours built on a subformula-filtered eval carries the
same caveat we are applying to theirs.
The same filter is applied to CASMI ("standardized and filtered to retain formula-supported peaks
before scoring"), but **not** to MassSpecGym, which "were evaluated using the provided test
spectra and annotations":

| Benchmark | Target peaks | Trained on it? | Reported mean cosine |
|---|---|---|---|
| NIST20 test | formula-filtered | yes | 0.83 (median 0.89) |
| CASMI 2022 | formula-filtered | no — external test only | 0.67 |
| MassSpecGym | as provided | no — external test only | 0.59 |

**Do not read the 0.83 -> 0.59 drop as the size of the filter effect.** "CASMI and MassSpecGym
were used only as external test benchmarks" (line 876): all three numbers come from the one
NIST20-trained model, so that drop confounds (a) the target filtering, (b) zero-shot domain
transfer to different instruments and collision energies, and (c) MSG's **MCES >= 10** separation,
which they enforced against their NIST *training* molecules. (c) is a far stronger generalization
demand than their own NIST test split, which only requires full-InChIKey disjointness plus
Tanimoto-1.0 removal, and could account for most of the gap by itself. Read the other way, 0.59
zero-shot under MCES >= 10 is arguably their strongest result — MSG leaderboard entries train on
MSG's own train split.

Consequences for us:
- **Do not compare 0.83 to our `cos_hun`** (per-CE CV 0.786, merged MASSter ~0.80). Ours is
  scored against raw experimental peaks. The comparison is not like-for-like and is not close to
  neutral — it favours their number.
- The *relative* ranking inside their paper is internally fair: all baselines were retrained and
  scored on the same processed splits (Fig. 3a: "All models were trained and tested with the same
  data splits, in both positive and negative modes"). Other models land at 0.61-0.75.
- MSG at 0.59 is **zero-shot transfer**, not an MSG-trained result, so it is not comparable to the
  MSG leaderboard in either direction and cannot be used to size the filter effect.
- To compare properly we would have to replicate the 5 ppm subformula filter and the
  "large unexplained peak" discard on our own test split, then rescore. Not yet done.

## NIST20 split

> "Spectra were split by **full InChIKey** into training, validation, and held-out test sets
> **before peak processing**, using an **80%, 10%, 10%** split, respectively." (line 869)

Other data decisions:
- Retained `[M+H]+` and `[M-H]-` only; excluded compounds containing metals; required each
  spectrum's structure to match the NIST20 formula and InChIKey connectivity.
- Structures canonicalized with RDKit tautomer enumeration.
- At evaluation, held-out targets identical to any training molecule at Morgan-Tanimoto 1.0 were
  removed. Default Morgan FPs are chirality-blind, so this recovers most of the leakage that
  splitting on the *full* (stereo-bearing) InChIKey would otherwise admit.
- MSG additionally drops test molecules with MCES distance < 10 to any NIST training molecule,
  applying the benchmark's own structural-separation criterion to their NIST-trained model.

### Versus our splits

| | ratio | split key | scope |
|---|---|---|---|
| AIMe | 80/10/10 | full InChIKey (stereo layer included) | `[M+H]+`, `[M-H]-`, no metals |
| GLACIER (`ms-pred` default) | 80/10/10 | InChIKey 2D, plus a Murcko scaffold variant | 530,640 HCD spectra |
| ours (`data/split/nist20_ma_mi_inchikey`) | 60/20/20 | `inchikey_s` (2D first block) | 4 adducts x FT/QTOF/IT, 480,002 spectra / 25,859 mols |

Our fractions come from [`preproc_scripts/04_prepare_split.py`](../../preproc_scripts/04_prepare_split.py)
(`--test_frac 0.2`, `--val_frac 0.2`, both taken off the total key count). Training on 60% vs
their 80% is a real handicap in any head-to-head; our 2x larger test set cuts the other way on
error bars.

## Retrieval protocol — it is ours

> "For inverse structure-identification evaluations, **we followed the candidate-selection
> procedure used by FraGNNet**. For each test spectrum, we identified candidate molecules whose
> precursor m/z was within 10 ppm of the query neutral monoisotopic exact mass and selected up to
> 49 candidates with the highest molecular-fingerprint Tanimoto similarity to the true structure
> as decoys." (line 888)

They note the m/z-based (rather than formula-based) selection admits near-isobaric non-isomers.
For CASMI they add a harder setting: all PubChem structures sharing the true formula (~2.5M
candidate spectra), where they report top-1/10/50 of 35.6 / 64.4 / 79.8% — decoy-hardness is not
the inflation risk there, but the cleaned CASMI targets still are. FraGNNet is the named runner-up,
beaten by >10% at top-1 and top-10.

## Method sketch

- **DeepMS2Reasoner** predicts a spectrum as a sequence of *fragmentation actions*, not a flat
  categorical over a pre-enumerated DAG. Symbolic operators enumerate chemically feasible
  bond-breaking transitions under hard constraints (hydrogen transfer, ring opening through
  allowed paths); a graph transformer, **ALEA** (Action Likelihood Estimator Advisor), assigns a
  normalized probability over the enumerated action set at each state.
- **No fixed maximum depth.** Pathways are expanded by probability mass, with canonical graph
  hashing merging equivalent states, so the DAG grows where the model puts probability. Contrast
  with our fixed d3/d4 enumeration — this is the same axis as
  `bond_breakability_mask_plan` (removed 2026-10-18; `git show 95341fce:docs/plans/fragnnet/bond_breakability_mask_plan.md`) (removed 2026-10-18; `git show 95341fce:docs/plans/fragnnet/bond_breakability_mask_plan.md`) fan-out control, solved by learned expansion instead.
- Markovian factorization over the DAG; KL-divergence loss against the experimental spectrum;
  curriculum learning from low-mass to high-mass compounds.
- Conditioned on global experimental metadata including collision energy.
- **MS2KOSMOS**: ~105M PubChem molecules x 4 NCEs (20, 30, 50, 80) x 2 ionization modes =
  >800M predicted spectra, stored as a sparse spectra-by-formula matrix ("Formula Coordinate
  System") with an inverted index from fragment formula to molecules, so retrieval only scores
  molecules sharing at least one fragment formula. Coarse retrieval by shared formulas, then
  re-ranking with fresh condition-matched predictions.
- Applied at repository scale to GNPS Living Data (8,543,020 clusters, 7,144,480 searched after
  filtering).

### No pretraining (checked 2026-09-27)

ALEA is "the only learned neural-network component" (line 803). It trains on the NIST20
80/10/10 split with the KL loss. The main text reports no pretrained initialization and no
training data beyond NIST20; CASMI and MassSpecGym are "used only as external test benchmarks"
(line 876). The closest thing to pretraining is the **molecular-mass curriculum**. Training
starts on lower-mass compounds, where the action space is smaller and pathways are shallower,
and adds heavier molecules as the action probabilities sharpen (line 858). They also apply
stochastic augmentation to the experimental metadata (line 863). **Not verified:** the
architecture specification is in "Supplementary AI Methods", which is not on disk, and the code
was not public as of 2026-09-20.

## Feature setup

Only the categories are published; "complete feature definitions, architectural specifications,
and decoder inputs are provided in Supplementary AI Methods", which is **not in the 44-page PDF**
we have. What the main text states:

- Feature-augmented **heavy-atom** graph. Hydrogens are not nodes: "hydrogen counts, hydrogen
  shifts and charge-related bookkeeping are encoded as atom-level features."
- Node- and edge-level features cover "atom identity, bond type, ring context, hydrogen
  bookkeeping together with global experimental metadata including **collision energy,
  instrument, adduct**" — note `instrument`, which GLACIER does not use.
- Graphormer with "learned structural bias terms derived from pairwise graph features"
  incorporated into self-attention.
- **Features are recomputed on the evolving fragment state**, since charge localization and
  hydrogen transfers are part of that state and later actions condition on it. This is the
  opposite choice from GLACIER, which computes molecule-level features once and reuses them
  across fragments; ours is molecule-level like GLACIER's.

For the feature comparison that has numbers behind it, see the GLACIER note — its Appendix A/C
tables are fully specified and are what our 2026-09-20 audit should be read against.

## Open questions for us

1. What does our test cosine become under their 5 ppm formula filter + unexplained-peak discard?
   That is the only way to read 0.83 against our numbers.
2. Their expansion is probability-mass-bounded rather than depth-bounded. Our measured ceiling
   work ([`dag_oracle_ceiling_nist20.md`](../findings/fragnnet/dag_oracle_ceiling_nist20.md)) says d3
   coverage is already high at 0.1 Da, so the gain would have to come from ranking, not support.
3. Formula-coordinate inverted indexing is directly reusable for our retrieval work regardless of
   the predictor.
4. Our coverage of their worked examples (Fig. 2c hydroxamic acid, Fig. 3k biotin) is checked in
   [`masster_smarts_findings.md#masster-smarts-templates-cuda-enablement-audit-and-proposal-reach` §6](../findings/masster/masster_smarts_findings.md#6-aimes-worked-examples-and-the-other-rule-lists).
   MASSter's plain cuts reach 14 of the 15 annotated fragments, and excision inside rings needs
   no rule for their masses. The miss is a hydroxamic-acid NH excision. The later §7 gate
   admitted its rule with weak evidence; §6 records the earlier rejection. These are mass
   reachability results, not verification of AIMe's product connectivity.
5. The [2026-09-28 excision literature follow-up](masster_excision_literature_2026_09_28.md)
   reviews experimental evidence and maps restricted candidates onto the current registry.
