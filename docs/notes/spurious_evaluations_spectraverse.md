# Confronting spurious evaluations / Spectraverse (Gupta & Skinnider)

> **Status:** NOTE &nbsp;·&nbsp; **Updated:** 2026-09-20
>
> Spectraverse is **this paper's benchmark**, and it is designed to remove the confounds that make
> MassSpecGym/PubChem retrieval numbers spurious. Read before interpreting any Spectraverse result
> or proposing a retrieval comparison.

- bioRxiv [10.64898/2026.05.03.722532](https://www.biorxiv.org/content/10.64898/2026.05.03.722532v1),
  posted 2026-05-06. Vishu Gupta & Michael A. Skinnider, Lewis-Sigler Institute / Ludwig Institute,
  Princeton.
- Spectraverse v1.0.2 on Zenodo: <https://doi.org/10.5281/zenodo.19927403> (spectra, generated
  candidate sets, CV fold assignments).
- Local copies (gitignored): `data/paper/Confronting_spurious_evaluations_biorxiv_2026.05.03.722532.{pdf,txt}`.
- **Our local data is this benchmark**: `data/proc/spectraverse/` has 488,797 spectra / 44,304
  molecules (paper: ">480,000 spectra from >44,000 compounds"), and
  `data/split/spectraverse_benchmark/cv0..cv9` are its ten specified folds.

## The structure-only model

A GNN trained to classify ground-truth vs candidate structures, **never seeing a spectrum**:
99.6% classification accuracy, AUROC 0.79 on held-out MassSpecGym.

| benchmark / split | structure-only top-1 |
|---|---:|
| MassSpecGym (MCES split) | 9.5% (top-5 27.7%, top-20 53.5%) |
| MassSpecGym, formula known (bonus) | 13.8% — "apparently state-of-the-art" |
| **MassSpecGym, conventional structure-disjoint split** | **26.9%** (top-5 54.5%, top-20 76.9%) |
| PubChem-derived candidate sets (22.4x / 6.8x larger) | still strong |
| **Spectraverse, generative decoys, mass-based** | **1.9%** vs 0.8% random |
| Spectraverse, generative decoys, formula-based | marginally above chance |

It **outperformed four of the five models MassSpecGym's own authors benchmarked** (best baseline
14.6% vs its 9.5%, but it beat the rest).

**Mechanism:** compounds in reference MS/MS libraries are structurally distinct from generic
chemical databases, so a model can learn "is this compound likely to have been measured by MS/MS"
instead of learning fragmentation.

**MOST IMPORTANT FOR US:** the confound is ~3x worse under a **conventional structure-disjoint
split** (9.5% -> 26.9%) than under MCES separation, because related structures may sit in
different folds. **That is exactly our split style** (`inchikey_s`, see
[our split ratios](../reference/dataloader_data.md#split-ratios-from-local-memory-re-measured-2026-10-06)). MCES-style separation suppresses it; ours maximises it.

**The fix:** train a generative model on reference-library structures and sample decoys from the
same region of chemical space. Residual above-chance performance traces to candidate sets where
only one generated structure is chemically plausible.

## Why Spectraverse is the benchmark to compete on

Its stated design points, each of which fixes something we hit elsewhere this session:

1. Ground-truth cannot be trivially separated from candidates without MS/MS information.
2. >480,000 spectra / >44,000 compounds.
3. Diverse ionization modes and adducts (our copy: 9 adducts, FT/QTOF/Other, **100% CID**).
4. **Ten CV folds, not a point estimate.**
5. **Training data is explicitly specified**, "to isolate the effects of modelling decisions from
   the composition of the training dataset ... given the well-documented pitfalls of incorporating
   auxiliary datasets." This is precisely the objection to the GLACIER comparison
   ([GLACIER gap decomposition](../findings/fragnnet/glacier_gap_ce_decomposition_2026_09_21.md)): GLACIER retrained baselines on its own split with numbers
   inherited from an unpublished ICEBERG 2.0, so nothing isolates modelling from data composition.
6. **Candidate sets are specified**, not left to each researcher — all models evaluated identically.
7. Large candidate sets act as a practical efficiency test.

**Usage rules from the paper:** train *exclusively* on Spectraverse's own ground-truth structures
and spectra (no auxiliary data), and report across **all ten** CV folds.

## Reading our own Spectraverse results

- The spurious-evaluation critique applies to **retrieval / annotation** (candidate ranking), not
  to **forward spectrum prediction**. Our `config/spectraverse/fragnnet_*` configs are forward
  prediction (cosine), where there are no candidate sets and the confound cannot operate.
- So a strong FraGNNet *retrieval* result on Spectraverse is **more** trustworthy than one on
  MassSpecGym or PubChem sets, not less — the shortcut is gone.
- A strong FraGNNet *forward* result is explained by measured factors, none of them artifacts of
  this paper's concern (measured 2026-09-20):
  - **merged target** — `merge: True`, 5.03 spectra/group; worth ~0.11 cosine on identical
    molecules per [merged-vs-unmerged root cause](../findings/masster/masster_unmerged_overfitting_2026_09_16.md);
  - **2.2x more training structures** — 33,947 train molecules vs 15,514 in
    `nist20_ma_mi_inchikey`;
  - **richer conditioning enabled only there** — ACE, frag_mode, ce_scale, frag_mode_scale,
    boundary_pair are live in spectraverse configs and off in the nist23 CV baseline
    ([`feature_set_audit.md`](../reference/feature_set_audit.md)).
- **Not an explanation: "easy precursor-dominated spectra."** Measured and refuted — Spectraverse
  test spectra have *more* peaks (median 23 vs 17) and *less* precursor dominance than NIST20, and
  a precursor-only prediction scores **lower** there (mean cosine 0.2054 vs 0.2487 on NIST20;
  base peak is the precursor in 15.6% vs 21.5% of spectra). Script:
  `scratchpad/precursor_baseline.py`.
- Unexplained and worth a look: Spectraverse is **100% CID**, historically FraGNNet's weakest frag
  mode on NIST (0.547 CID vs 0.781 HCD per-CE CV). A CID-only benchmark scoring well is the
  opposite of what that breakdown predicts.
