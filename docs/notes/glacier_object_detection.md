# GLACIER (Wang, Wang & Coley, MIT — arXiv 2606.29161)

> **Status:** NOTE &nbsp;·&nbsp; **Updated:** 2026-09-20
>
> Literature notes on "GLACIER: Rethinking Mass Spectrum Prediction as an Object Detection
> Problem". Read the **bin-width caveat** before quoting any of its NIST'20 numbers.

- arXiv: [2606.29161](https://arxiv.org/abs/2606.29161) · code in
  [`coleygroup/ms-pred`](https://github.com/coleygroup/ms-pred) (`src/ms_pred/glacier/`)
- Local copies (gitignored): `data/paper/GLACIER_arxiv_2606.29161.pdf` and the extracted `.txt`
  beside it. Line numbers below refer to the `.txt`.
- Companion note: [`aime_deepms2reasoner.md`](aime_deepms2reasoner.md). Same shape of protocol
  problem in a different place — GLACIER scores at a **bin width 100x coarser than the
  instrument**, AIMe cleans the **target spectra**. Neither set of numbers can sit beside ours
  without redoing the protocol.

## Bin-width caveat — READ FIRST

> "Binned spectral vectors are used with **0.1 resolution on the NIST'20 dataset** and **0.01
> resolution on the MassSpecGym dataset** to align with prior work on both benchmarks."
> (Section 4.1, line 445)

Every NIST'20 number in Tables 15-17 — cosine 0.838, entropy 0.781, coverage 0.887 — is at
**0.1 Da on Orbitrap data**, i.e. ~333 ppm at m/z 300. A single 0.1 Da bin merges CO (27.9949),
N2 (28.0062) and C2H4 (28.0313), so the standard neutral-loss ambiguities collapse and
fragment-identity error is invisible to the metric. The paper itself flags the hazard when it
belongs to someone else: MolSpecFlow "uses a different bin width (0.1 Dalton) instead of the bin
width adopted by MassSpecGym (0.01 Dalton); the comparison has to be made with extra care."

**The choice lands on the flattering side of each metric.** Their own Appendix F (Table 14,
MassSpecGym top-1 retrieval):

| train bin | test @ 0.1 Da | test @ 0.01 Da |
|---|---|---|
| 0.1 | 0.664 mass / 0.479 formula | **0.685 / 0.490** |
| 0.01 | 0.652 / 0.471 | **0.697 / 0.485** |

Retrieval is *better* at 0.01 Da (finer bins discriminate decoys); cosine moves the other way
(coarse bins forgive m/z error). So NIST'20 reports **cosine + coverage at the forgiving 0.1 Da**
while MassSpecGym reports **retrieval at the discriminating 0.01 Da**. Whether or not that is
deliberate, both reported metrics get the bin width that helps them.

## NIST'20 results (Appendix G, Table 15, `[M+H]+`, 0.1 Da)

| Method | random split cosine | scaffold split cosine | coverage (random) |
|---|---|---|---|
| CFM-ID | 0.470 | 0.450 | 0.261 |
| GrAFF-MS | 0.565 | 0.470 | 0.740 |
| MassFormer | 0.632 | 0.526 | 0.787 |
| **FraGNNet** | **0.717** | **0.654** | - |
| ICEBERG 1.0 | 0.754 | 0.701 | 0.809 |
| ICEBERG 2.0 | 0.794 | 0.735 | 0.856 |
| GLACIER w/o CF | 0.838 | 0.788 | 0.887 |
| GLACIER w/ CF | 0.815 | 0.763 | 0.858 |

- **Our number in that table is not our protocol.** "Baseline results are adapted from Wang et al.
  [29]" (the ICEBERG 2.0 paper) — a third-party re-evaluation at 0.1 Da, with entropy and coverage
  left blank. Our own reported cosines are 0.01 Da binned (`val_cos_sim_0.01`) or
  instrument-tolerance Hungarian (`cos_hun`). Do not place our numbers next to this column.
- Contrastive finetuning *lowers* spectral similarity (0.838 -> 0.815) while raising retrieval —
  consistent with CF trading calibration for discriminability.

## Coverage is top-20 at 0.1 Da

Confirmed in the released evaluator, [`analysis/spec_pred_eval.py`](https://github.com/coleygroup/ms-pred/blob/main/analysis/spec_pred_eval.py):
`--max-peaks` defaults to 20 (line 48), and coverage is
`|top-20 true bins ∩ top-20 predicted bins| / |top-20 true bins|` (lines 119, 167-170). Combined
with 0.1 Da, "coverage 0.887" asks: of the 20 largest 0.1-Da target bins, how many are among the
20 largest predicted bins. Our d3 DAG oracle at 0.1 Da is 0.9334 and d4 is 0.9475 (see
[`dag_oracle_ceiling_nist20.md`](../findings/dag_oracle_ceiling_nist20.md)), both above their
achieved 0.887 — so under this protocol our candidate support is not the binding constraint.
Still unconfirmed: whether the *paper's* runs used the evaluator default of 20.

## Split

The paper gives no ratios. It states only that it evaluates a random split "partitioned by InChI
key" and a Murcko scaffold split, over 530,640 HCD spectra / 25,541 unique structures. The repo
default (`data_scripts/make_splits.py`) is `--train-frac 0.8 --val-frac 0.1 --test-frac 0.1`
applied at the object level, which measures 0.809/0.090/0.101 on the checked-in `split_1.tsv`.
Ours is 60/20/20 — see `[[project_split_ratio_ours]]`.

## Method

Single-stage fragment "object detection": a transformer predicts fragment subgraph masks directly
from the molecular graph, replacing the enumerate-then-score two-stage pipeline (their own
ICEBERG). Affinity matrix -> binary subgraph masks with a temperature-controlled relaxation;
binary attention masks restrict each fragment's attention to its own atoms. Losses: cosine
spectral distance plus an auxiliary loss on heuristically-labeled fragment patterns. Claims ~8x
inference speedup over the two-stage model. Intensities are predicted unbinned and re-binned at
inference, which they argue lets one model serve multiple bin widths.
