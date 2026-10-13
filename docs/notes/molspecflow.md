# MolSpecFlow (Peking U / Tencent, bioRxiv 2026.01.28.702438)

> **Status:** NOTE &nbsp;·&nbsp; **Updated:** 2026-09-20
>
> Literature notes on "MolSpecFlow: Mass-Constrained Hybrid Flow Matching for Joint
> Molecular-Spectral Analysis". **No code released**; its simulation numbers are at 0.1 Da.

- bioRxiv [10.64898/2026.01.28.702438](https://www.biorxiv.org/content/10.64898/2026.01.28.702438v1),
  posted 2026-02-01. Wang, Yang, Xu et al.; corresponding Jianhua Yao, Yonghong Tian, Fan Yang.
  Peking University / Tencent AI for Life Sciences Lab / CAMS, with Tsinghua, CUHK, NUS.
- Local copies (gitignored): `data/paper/MolSpecFlow_biorxiv_2026.01.28.702438.{pdf,txt}`.
- Companion notes: [`glacier_object_detection.md`](glacier_object_detection.md),
  [`aime_deepms2reasoner.md`](aime_deepms2reasoner.md).

## What it is

A **bidirectional** foundation model (molecule <-> spectrum), not a spectrum predictor. Method is
"mass-constrained hybrid flow matching": optimal-transport flow paths for spectral peaks
(continuous) plus discrete probability flows for molecular tokens, with a token-level mass control
mechanism forcing generated structures to match the precursor mass. **Pretrained on 100M molecules
and 42M spectra**, then fine-tuned on MassSpecGym. Reports a neural-scaling curve over
0M/1M/10M/100M pretraining.

Evaluated only on MassSpecGym, using the official MCES split (edit distance >= 10), on three
tasks: de novo generation, spectrum simulation, retrieval.

## Results

**De novo generation (Table 1)** — the abstract's "+35%" is RELATIVE (3.11% vs 2.30%):

| model | Top-1 | MCES | Tanimoto | Top-10 |
|---|---:|---:|---:|---:|
| MADGEN (pred.) | 1.31% | 27.47 | 0.20 | 1.54% |
| DiffMS | 2.30% | 18.45 | 0.28 | 4.25% |
| **MolSpecFlow (pred.)** | **3.11%** | 17.32 | 0.34 | 5.23% |
| MADGEN (oracle) | 10.5% | 16.27 | 0.43 | 12.4% |
| **MolSpecFlow (oracle)** | **13.6%** | 16.01 | 0.54 | 17.1% |

They acknowledge the low absolute Top-1 as expected on MCES splits.

**Spectrum simulation (Table 2), binned at 0.1 Da:**

| model | cosine | J-S | HR@1 |
|---|---:|---:|---:|
| Precursor m/z | 0.15 | 0.15 | 0.38 |
| GNN | 0.19 | 0.20 | 3.95 |
| FFN fingerprint | 0.25 | 0.24 | 8.44 |
| **FraGNNet** | **0.52** | 0.47 | 46.64 |
| **MolSpecFlow** | **0.63** | 0.59 | 55.32 |

**Retrieval vs MIST (Table 3):** MolSpecFlow 15.31 / 35.76 / 61.32 (HR@1/5/20), MCES@1 14.32;
MIST 14.64 / 34.87 / 59.15, MCES@1 15.37. **Only +0.67 points at HR@1** — far weaker than the
simulation table suggests.

## Caveats

1. **Their simulation numbers are at 0.1 Da** (Appendix E.3: "Both spectra are binned
   (bin size = 0.1 Da) and normalized"), while GLACIER's MassSpecGym numbers are at 0.01 Da.
   MolSpecFlow's 0.63 and GLACIER's 0.58 are **not comparable**, and MolSpecFlow never flags it.
   GLACIER does: "It also uses a different bin width (0.1 Dalton) instead of the bin width adopted
   by MassSpecGym (0.01 Dalton); the comparison has to be made with extra care."
2. **The retrieval mechanism is ambiguous between their own tables.** Appendix E.4 says retrieval
   works by predicting an ECFP fingerprint from the spectrum and ranking candidates by Tanimoto —
   the MIST/CSI:FingerID mechanism, matching Table 3. But Table 2's HR column sits beside
   simulation metrics with simulation baselines. The paper never says which mechanism produced the
   **55.32** that GLACIER cites in its mass-challenge table. Do not repeat 55.32 without this.
3. **No code availability statement anywhere in the preprint.** GLACIER confirms: "not
   open-sourced", and no speed statistics are reported, so it cannot be benchmarked.
4. The FraGNNet HR@1 of 46.64 is identical in both papers — it is the MassSpecGym-published
   number, not a re-run by either group.

## Relevance to us

- Its real contribution is **joint generation + pretraining scale** (100M molecules / 42M spectra),
  not forward spectrum prediction. Different competitive axis from the fragment-based line
  (ICEBERG / FraGNNet / MARASON / GLACIER).
- On simulation it is a 0.1 Da number with no code, so it is not a reproducible target.
- Its FraGNNet simulation number (0.52 cosine on MSG at 0.1 Da) is a third-party evaluation of our
  model — same caution as GLACIER's Table 15 row (see the GLACIER note).
