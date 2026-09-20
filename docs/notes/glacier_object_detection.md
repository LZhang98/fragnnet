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

## Dataset scale and m/z limit

**They do not have more data than us — the difference is merged vs unmerged.**

| | spectra | molecules | spectra/mol |
|---|---:|---:|---:|
| GLACIER NIST'20 | 530,640 | 25,541 | 20.8 |
| ours `nist20_ma_mi_inchikey` | 480,002 | 25,859 | 18.6 |
| ours, merged | ~26,000 | 25,859 | 1.01 |

Molecule counts are within 1.2%. The 20x impression comes from comparing their unmerged count
against our merged setup (1.01 rows/mol, see `[[project_masshatter_merged_vs_unmerged_root_cause]]`).
The residual 10% is composition: theirs is all HCD in both polarities, ours spans FT/QTOF/IT over
4 adducts. Against the older ms-pred NIST20 subset (35,129 spectra in the checked-in
`split_1.tsv`, the ICEBERG 1.0 / SCARF era) GLACIER is ~15x larger — a change in extraction, not
a new data source.

**m/z limit is 1500 Da**, the same as ours. Not stated in the paper; from
`coleygroup/ms-pred@main` (verified upstream, not just the old local checkout):
`bin_spectra(spectras, num_bins=15000, upper_limit=1500, pool_fn="max")`, with the same defaults
in `MassSpec.ensure_binned_spectrum`, `_bin_spec_sparse` and `PredSpecDB.bin_spectrum`.
15,000 bins over [0, 1500) **is** the 0.1 Da resolution. Ours: `--max_peak_mz 1500.0`,
`--max_prec_mz 1500.0` (live in `data/split/nist20_ma_mi_inchikey/meta.json`).

Note the ceiling and the resolution are one constant pair: at a fixed 1500 Da ceiling, 0.01 Da
means 150,000 bins per spectrum instead of 15,000, a 10x blow-up across 530,640 spectra. That is
a plausible compute reason for leaving NIST'20 at 0.1 Da while MassSpecGym (231,104 spectra) got
0.01 — **inference, not stated in the paper**. It does not change that the reported cosine
benefits from the coarser bin.

## Feature setup (Appendix A, Tables 6-7; Appendix C)

Graphormer over the heavy-atom graph.

- **Atom**: type one-hot + "other"; **degree one-hot (excl. H) + overflow**; formal charge one-hot
  + overflow; hybridization one-hot + other; aromaticity; H-count one-hot + overflow; ring
  membership; **ring-size multi-hot**; **fused-ring**; chirality one-hot; optional element-group
  embedding; **Gasteiger charge**.
- **Bond**: type one-hot + other; conjugation; ring membership; **ring-size multi-hot**;
  **fused-ring**; stereo one-hot + other.
- **Metadata**: adduct one-hot and collision energy sinusoidally encoded, projected into *both*
  node embeddings and edge-level attention biases (Section 3.1). No instrument feature.
- RDKit features are computed **once at molecule level and reused across fragments** — pitched
  explicitly as an advantage over ICEBERG's per-fragment embeddings.

**Intensity head inputs (Appendix C, eq. 14)** — the part that matters most for us:

```
t_k = MLP( G || psi(f_k) || psi(r_form - f_k) || OHE(n_broken_k) )
```

fragment formula, the **neutral-loss complement** (root formula minus fragment formula), and the
**broken-bond count**, with `psi` a sinusoidal formula embedder and `G` the root-molecule token.
That is exactly the two design gaps in our 2026-09-20 feature audit: our candidate intensity head
is mass- and composition-blind while `ion_mzs` is already in scope, and the cut bonds reach
intensity scoring only through the GNN.

### Versus our MASShatter feature set

All 167 `model_type: mass_shatter` configs carry an identical set: `a_onehot, a_degree, a_hybrid,
a_formal, a_radical, a_ring` (ring + aromatic), `a_mass, a_chiral, a_ring_size, a_num_hs`; bonds
`b_degree` (type), `b_ring`, `b_conj`.

| GLACIER has | us |
|---|---|
| Degree one-hot + overflow | `a_degree`'s 12 one-hot columns were dead (list-vs-int); repaired in place 2026-09-20 at unchanged width |
| Fused-ring, atoms and bonds | absent |
| Bond ring-size multi-hot | atoms only (`a_ring_size`) |
| Bond stereo | `b_stereo` registered, unused |
| Gasteiger charge | `a_charge` registered, off in every MASShatter config |
| - | we add `a_mass`, `a_radical` |

Note the `feature_set_audit` "gaps `a_ring_size`, `b_ring`" line is **template-only** — MASShatter
configs enable both, plus `b_conj` and `a_num_hs`. `config/*_template.yml` still has them commented.

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
