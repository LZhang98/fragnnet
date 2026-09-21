# MassSpecGym in the Wild — evaluation audit (arXiv 2606.19624)

> **Status:** NOTE &nbsp;·&nbsp; **Updated:** 2026-09-20
>
> "Uncovering and Correcting Evaluation Pitfalls in AI-Driven Molecule Discovery". **Read before
> designing or citing any MassSpecGym comparison.** Evaluation issues found in **17 of 26** papers
> reporting MassSpecGym results in the benchmark's first year.

- arXiv [2606.19624](https://arxiv.org/abs/2606.19624). Liu, R. Bushuiev, Lightheart, Manjrekar,
  A. Bushuiev, Lederbauer, Jozefov, Y. Wang, Hassoun, Sivic, Taylor, R. Wang, Healey, Pluskal,
  Coley. MIT / IOCB Prague / CIIRC CTU / Enveda Biosciences / Tufts.
- Benchmark revision released as **MassSpecGym v1.5**, <https://github.com/pluskal-lab/MassSpecGym>.
- Local copies (gitignored): `data/paper/MassSpecGym_in_the_Wild_arxiv_2606.19624.{pdf,txt}`.
- **We are already on v1.5** — `data/split/ms_gym_benchmark_1_5` and `ms_gym_all_1_5` use the
  official predefined split over `ms_gym_benchmark_1_5` + `ms_gym_extra_1_5`.

## The corrected retrieval leaderboard (Table 3, formula-known track)

| method | HR@1 |
|---|---:|
| **(3) ChemBERTa SMILES classifier — NO spectrum, format only** | **99.66%** |
| (3) same, corrected canonicalization | 8.2% |
| (1) DreaMS + ChemBERTa alignment — inflated / corrected | 82.41% / 6.70% |
| (2) ChemFormer two-stage — inflated / corrected | 63.89% / 7.60% |
| **(4) PubChem default ranking — spectrum-blind** | **49.98%** |
| FRIGID (generative retrieval) | 45.27% |
| MIST vDiffMS | 46.42% |
| MIST+MolForge (generative retrieval) | 39.09% |
| ICEBERG (forward model retrieval) | 36.18% |
| DiffMS (generative retrieval) | 32.29% |
| FLARE | 22.66% |
| MIST (MassSpecGym version) | 9.57% |
| (5) ranking by num. chiral atoms | 4.88% |

**A classifier trained only on SMILES syntax reaches 99.66%** — ground-truth strings kept their
source-repository formatting while decoys were RDKit-canonicalized, so the true molecule is the
one formatted differently. **Ordering candidates by PubChem's default API ranking, never looking
at the spectrum, beats every published model in the table.**

## Three failure classes

1. **Data leakage (§4).**
   - §4.1 de novo decoders pretrained to decode fingerprints: accuracy correlates with Tanimoto
     similarity to the *pretraining* set. Fix: exclude test structures **and close analogs
     (Tanimoto >= 0.7)** from pretraining. v1.5 ships 2.5M and 50M safe corpora.
   - §4.2 spectrum-to-fingerprint encoders pretrained on external data memorise test examples.
     Transitive leakage through an ICEBERG used for synthetic augmentation: MIST retrieval goes
     0.262 -> 0.640 cosine / 9.57% -> 67.75% HR@1 across leakage settings.
   - §4.3 **implicit formula leakage in the mass-based track.** Pruning candidates to the
     ground-truth formula lifts a *random* baseline from 0.43% to 12.98% HR@1 (30x). A test-
     contaminated MIST-CF formula annotator adds up to +9 points. Methods that assume known
     formula must report on the formula-based track.
2. **Shortcut learning (§5).** SMILES canonicalisation (§5.1, fixed in v1.5) and **PubChem
   ranking bias (§5.2, NOT fixed)** — the authors call it "an open design problem inherent to the
   retrospective nature of this, or any, benchmark."
3. **Implementation bugs and metric divergence (§6).** MIST inference batching pitfall; and four
   metrics with unstandardised implementations. Pinned choices: **14-character** InChIKey
   connectivity layer (not the 27-char key), **2048-bit ECFP4** Tanimoto, MCES `threshold=15`
   with `always_stronger_bound=True`, and **99.9% CI over 20,000 bootstrap resamples**.

## Do people report MassSpecGym correctly? No — ~2/3 do not

34 papers used MassSpecGym in year one (cutoff 2026-05-01); 8 used it as a *dataset*, not a
benchmark. Of the **26 reporting it as a benchmark, at least 17 have evaluation issues** that
affect the validity of their conclusions (**65%**).

The two most mechanical illustrations:

**(a) Identical predictions, different reported numbers** (Table 10, same DiffMS outputs):

| metric variant | Top-1 |
|---|---:|
| Tanimoto Morgan r=2, 1024b | 0.2968 |
| **Tanimoto Morgan r=2, 2048b (reference)** | **0.2857** |
| Tanimoto Morgan r=3, 2048b | 0.2209 |
| Tanimoto Morgan r=3, 4096b | 0.2139 |
| MCES thld=100, bound=False | 21.21 |
| **MCES thld=15, bound=True (reference)** | **16.02** |

A **39% relative spread in Tanimoto** and 34% in MCES from undocumented fingerprint/solver
parameters alone. (Accuracy was identical for InChIKey-14 vs full key on this model.)

**(b) A one-line bug inherited through checkpoint reuse** (§6.1). MIST applied its padding mask as
`attn += attn_mask`, giving padding tokens a finite +1 attention offset instead of excluding them
from the softmax — affecting training *and* batched inference. The resulting MIST (vDiffMS)
checkpoint was reused as a frozen feature extractor in several de novo pipelines:

| MIST + MolForge | Top-1 | MCES | Tanimoto |
|---|---:|---:|---:|
| corrected | 10.73% | 22.15 | 0.37 |
| **inflated (batch size 24)** | **31.75%** | 12.30 | 0.68 |

Nearly **3x inflation**. The auditors add that the inflation "is consistent with patterns of data
leakage or contamination; however, we cannot reproduce this inflation from the documented training
recipe" — the corrupted checkpoint's origin is unexplained.

The dominant causes are **inherited components and unpinned metric parameters**, not deliberate
optimism. Both are controllable in our own pipeline.

## What this means for us

- **Do not target the MassSpecGym retrieval gap as a "clean public benchmark."** It was proposed
  in this session as the argument-free target because FraGNNet sits at 46.64 HR@1 against
  GLACIER's 69.95 on the simulation track's mass challenge. That was wrong: a spectrum-blind
  PubChem ordering scores 49.98% on the retrieval challenge, and the candidate-set ranking bias
  is explicitly unfixed in v1.5.
- **Any retrieval number we publish needs spectrum-blind controls next to it** — at minimum
  PubChem default ranking, ideally the chiral-atom ranker — or a reviewer can't tell whether we
  measured spectral reasoning.
- **If we ever pretrain on external molecules**, exclude Tanimoto >= 0.7 analogs of test
  compounds, or use v1.5's 2.5M/50M corpora.
- **Pin the metric implementations** above for any MassSpecGym comparison; 14-char InChIKey in
  particular, since our `inchikey_s` split key is already the 2D first block (consistent).
- Relevant to the competitive read: **MolSpecFlow pretrains on 100M molecules / 42M spectra** and
  reports de novo generation — precisely the §4.1 risk class, unaddressed in its preprint. See
  [`molspecflow.md`](molspecflow.md).
- The audit covers the benchmark's first year and does **not** name GLACIER or MolSpecFlow; both
  postdate it. Absence from the audit is not a clean bill of health.
