# Why we are not at 0.838: the GLACIER gap is CE-dependent allocation

> **Status:** FINDING · **Updated:** 2026-09-21
> The 0.077 gap to GLACIER on `[M+H]+` is entirely at high collision energy and is an allocation deficit, not coverage; realizing our low-CE rate uniformly would give 0.835.

## The like-for-like number

GLACIER reports **0.838** on NIST'20 `[M+H]+`, random split, **0.1 Da bins**, per-CE (their data is
20.8 spectra/molecule, i.e. unmerged).

Ours, `pqx7ywc9` (FraGNNet), FT / `[M+H]+` test slice of `nist20_ma_mi_inchikey`, 0.1 Da bins,
sum-pooled predictions, n=57,172: **0.7612**. Gap **0.077**.
The slice is 92.8% HCD / 7.2% CID, so frag-mode contamination is not a factor. HCD NCE
distribution: median 50, p75 90, p90 130, max 195.

## The gap is entirely at high CE

FT / `[M+H]+` / **HCD** only (n=53,058). Oracle is the DAG ceiling from
`data/eval/oracle_cos_nist20_default/oracle_cos_default.csv` (`cos_hun_opt`).

| NCE | n | DAG oracle | peak coverage | model @0.1 Da | **% of oracle realized** |
|---|---:|---:|---:|---:|---:|
| 0-20 | 6,625 | 0.9961 | 0.828 | 0.8811 | **88.4%** |
| 20-40 | 12,473 | 0.9863 | 0.830 | 0.8000 | 81.1% |
| 40-60 | 10,425 | 0.9728 | 0.797 | 0.7765 | 79.8% |
| 60-80 | 7,636 | 0.9476 | 0.746 | 0.7331 | 77.4% |
| 80-100 | 4,587 | 0.9179 | 0.718 | 0.7037 | 76.6% |
| 100-120 | 4,843 | 0.8878 | 0.695 | 0.6763 | 76.2% |
| 120+ | 6,469 | 0.8618 | 0.686 | 0.6515 | **75.6%** |
| **overall** | 53,058 | **0.9492** | 0.772 | **0.7612** | **80.2%** |

- **At NCE < 20 we already beat GLACIER** (0.8811 vs 0.838). At NCE < 40 we are at 0.8231.
  The 28% of spectra above NCE 80 score 0.65-0.70 and account for the whole deficit.
- **Coverage is not binding.** The oracle degrades with energy (0.996 -> 0.862) but never falls
  below 0.86; a perfect allocator would beat 0.838 in every bucket.
- **What degrades is realization**: 88.4% of the achievable ceiling at low CE, 75.6% at high CE.

**The arithmetic that defines the target: 0.884 x 0.9492 = 0.835.** Realizing the low-CE rate
uniformly reaches GLACIER's number with **no DAG change, no deeper fragmentation, no extra
training data.** The entire gap is CE-dependent intensity allocation.

## How to get there, ranked

1. **Marginal sibling-ranking loss on P(n)** (sibling-ranking plan (removed 2026-10-18; `git show 95341fce:docs/plans/fragnnet/sibling_ranking_target_plan.md`), GREEN,
   583,539 clean targets at margin 0.1). A ranking/allocation fix, and
   the branching headroom probe ([FluxMPNN plan](../../plans/fragnnet/fluxmpnn_plan.md)) independently measured that observed sibling branching is
   strongly CE-dependent (same-NCE floor 0.11 -> 0.68 at max NCE gap) — the same mechanism reached
   from a different direction. **Run first.**
2. **Composition-aware intensity head** — fragment formula + neutral-loss complement +
   broken-bond count (GLACIER eq. 14; see [`notes/glacier_object_detection.md`](../../notes/glacier_object_detection.md)).
   Our candidate head is blind to all three, and these are exactly the signals that discriminate
   competing candidates.
3. **Intensity-head capacity on the per-CE arm** — 384x3/FF1024/ind128/ad4 + `independent` +
   `jss_distance`; never varied unmerged across 31 runs, worth +0.0129 epoch-matched on merged.
4. **Coverage at high CE** (d4 / ring-merge) — secondary. It raises a ceiling we realize only 76%
   of, so it is worth less than any allocation fix until realization improves.
5. **80/10/10 split** — removes the training-data handicap, but is clearly not the main story.

## Open tension — reconcile before betting on CE conditioning

the [CE-extent null](ce_extent_headroom_finding.md) (CE -> "mass moves deeper" has no headroom; error is
CE-invariant) and the FluxMPNN Stage 0 gate ([plan](../../plans/fragnnet/fluxmpnn_plan.md)) (dominance mismatch is CE-FLAT,
not high-NCE-concentrated) both concluded the error is CE-invariant. The realization ratio here
falls monotonically with CE. These measure different quantities — fragmentation extent and
sibling-dominance mismatch vs overall cosine realization — but the conflict should be resolved
before committing to a CE-conditioned module, since one of those findings says that axis is dead.

## Provenance

Scripts: `scratchpad/ce_decomp.py` (model, 0.1 Da, sum-pooled, NCE-stratified) and the oracle join
against `data/eval/oracle_cos_nist20_default/oracle_cos_default.csv`. Bin-width and pooling
conventions replicate `ms_pred.common.bin_spectra` — see
[`notes/glacier_object_detection.md`](../../notes/glacier_object_detection.md).
