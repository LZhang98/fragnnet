# Cross-adduct loss: data-side headroom gate

> **Status:** FINDING · **Updated:** 2026-09-17 · Model-free measurements on NIST20 only.
> The data-side gate passes; the trained-model half of the gate is still open and blocked
> on a multi-adduct MASSter checkpoint.

## Question

Is it worth adding a loss that groups MS/MS of the same molecule across precursor adducts and
supervises them jointly? Two ways it could be worthless: too few matched pairs, or a difference
between adducts that is a global, molecule-independent pattern the existing adduct embedding
already gets for free. This is the same failure mode as the closed CE-to-extent allocation line.

## Result

The gate passes. Matched pairs are plentiful, split-safe, and the cross-adduct difference is
about **92% molecule-specific**.

### Pair availability

NIST20, the four arm adducts on FT/QTOF: 580,283 rows over 30,935 molecules. **46.5% of
molecules (14,370) carry at least two adducts.**

| Matched on | Groups with >= 2 adducts | Rows covered |
| --- | --- | --- |
| Molecule only | 16,963 | 65.5% |
| Molecule + instrument + fragmentation mode | 26,777 | 64.2% |
| Molecule + instrument + fragmentation mode + NCE | 78,938 | 31.9% |

In the `nist20_ma_mi_inchikey` split itself: 45,426 multi-adduct groups covering 104,725 train
rows (22.1% of train), and 15,530 groups covering 35,622 validation rows. Co-occurrence is
dominated by `[M+H]+`/`[M-H]-` (20,919 groups at molecule + instrument + fragmentation mode),
then `[M+H]+`/`[M+Na]+` (9,898) and `[M+Na]+`/`[M-H]-` (7,731).

**Split-safe:** 0 of 25,859 molecules span train/val/test, because the split is by InChIKey. An
adduct group is therefore always inside one split; no leakage, and the term is computable on
validation as a monitor.

### Shared structure is large but partial

Same molecule, `[M+H]+` versus `[M-H]-`, matched on instrument, fragmentation mode and NCE,
peaks converted to neutral-equivalent mass at 5 mDa, top 20 peaks, 3,000 pairs:

| | Matched intensity |
| --- | --- |
| Same molecule | 0.404 mean, 0.330 median |
| Different molecule (control) | 0.014 mean, 0.000 median |
| Pairs above 50% matched | 38.5% versus 0.9% |

About 40% of intensity is shared neutral-fragment structure at 29x over control, and about 60%
is adduct-specific. That rules out a penalty that forces `P(n)` to be equal across adducts, and
supports a shared-subset or delta formulation instead.

### The difference is not a global adduct pattern

Neutral-loss space, FT/HCD, matched `(molecule, NCE)` pairs, 0.01 Da bins, top 40 peaks. A
molecule-independent reweighting `w(NL)` was fitted on 4,147 molecules and evaluated on a
**disjoint** 4,147 molecules (25,214 held-out pairs, 6,682 replicate pairs):

| | Cosine |
| --- | --- |
| Replicate floor, same molecule/adduct/NCE | 0.950 |
| Base, `cos(+H, -H)`, no correction | 0.259 |
| Global, `cos(+H * exp(w_NL), -H)` | 0.323 |

The global pattern closes **9.3%** of the 0.691 gap to the replicate floor; **90.7% is
molecule-specific**. Conditioning the global model further on NCE and precursor-mass buckets
made it slightly *worse* on held-out molecules (0.313, 7.8% closed) by overfitting sparser
contexts, so the conclusion is not an artifact of an under-powered null.

## Limits

- Model-free. This establishes that the effect is large, structured and not a fixed global
  transform. It does **not** establish that a trained model fails to reproduce it: MASSter's
  adduct conditioning enters the context and modulates the encoder over the molecule graph, so
  it is molecule-dependent and strictly more expressive than the `w(NL)` null used here. What a
  cross-adduct loss adds is *paired* supervision coupling the two spectra, which is a different
  signal from learning each adduct independently. Whether joint training beats independent
  training is the open half of the gate.
- The remaining half needs a trained multi-adduct MASSter checkpoint. Both checkpoints
  present in this checkout (`zlv0bli8`, `m14g0j35`) are `[M+H]+` only, so it could not be run
  here. The `ma_mi_unmerged/cmf` arms are the ones with all four adducts.
- Neutral-equivalent mass uses a single proton shift per polarity. It does not model the full
  H-transfer ladder or `[M+Na]+`/`[M+Cl]-` deltas, so it undercounts matches involving
  multiple H transfers. Only `[M+H]+`/`[M-H]-` was measured in the neutral-loss probe.
- Top-N peak truncation (20 or 40) and fixed binning were not swept.

## Reproduce

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python scripts/debug/masster_candidates/probe_cross_adduct_headroom.py
```

Requires `data/proc/nist20/spec_df.pkl`. Runs on CPU in a few minutes; no model or checkpoint.

## Next

Run the trained-model half against a multi-adduct checkpoint: measure the cross-adduct fragment
intersection under the real top-256 proposals, and whether the model's predicted cross-adduct
delta already matches the observed one. Only then implement. The chosen formulation is a delta
loss on the shared fragment subset, keeping adduct-conditioned proposals; note that
[masster_model.py](../../src/fragnnet/model/masster_model.py) asserts a single
precursor type per shared-proposal group, and `group_id` today never spans adducts
(0 of 76,164 groups), so both need a second grouping key.
