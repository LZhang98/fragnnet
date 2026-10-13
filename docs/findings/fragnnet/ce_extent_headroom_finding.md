# CE → fragmentation-extent allocation: measured null

> **Status:** CLOSED NULL · **Updated:** 2026-07-16
> The CE -> 'mass moves deeper' prior has no headroom. Kills the CFM-flow stop head, `depth_ce_bias`, and a CE x `num_cut_bonds` bias. Read before building any CE-conditioned module.

**Date:** 2026-07-16
**Script:** `scripts/eval/ce_extent_headroom.py`
**Status:** CLOSED — kills the CFM-flow stop head, `depth_ce_bias`, and a CE × `num_cut_bonds` bias, with zero GPU spend.

## The question

Several proposed mechanisms encode the same prior:

> as collision energy rises, intensity should move from the precursor toward smaller fragments.

That prior is the *entire* mechanism behind the CFM-style flow head's CE-conditioned stop
probability, and it is also what `depth_ce_bias` (`fragnnet_model.py:513-524`) and a CE ×
`num_cut_bonds` bias would encode additively.

Before building any of them, two things must be true:

1. the trend exists in the data, and
2. the model does not already capture it.

Both are measurable from `spec_df` + the DAG with no model training. This is what the script does.

## Method

### Coordinate: retained heavy-atom fraction

```
retained_frac(formula) = heavy_atoms(formula) / heavy_atoms(precursor)
```

1.0 is the intact precursor; small values are heavily fragmented.

**Why not DAG depth.** Depth is not a fragmentation-extent proxy in this codebase. The SMARTS
ring-cleavage rules break 4 bonds in a single BFS step, so a 4-cut xanthone node and a 1-cut node
are both depth 1 (see the `num_cut_bonds` comment at `fragnnet_model.py:1174-1178`). The `depth`
node feature is also a *multi-hot bitmask* of every depth a fragment was reached at
(`multi_cut_bfs.pyx:919-927`), not a scalar. Retained heavy-atom fraction is a property of the
formula alone: unaffected by H-shifts, needs no node→formula attribution, well defined for both
the true and the predicted spectrum. The denominator is the max heavy-atom count over the DAG's
formulas, which is the root by construction — so the coordinate is self-contained.

### Attribution and its ambiguity

A true peak can match several DAG formulas of different heavy-atom counts. Rather than pick one,
every statistic is reported as a **bracket**: `min` attributes each peak to the smallest matching
formula, `max` to the largest. Both the true and the predicted spectrum go through the *identical*
DAG-ion matching (CMF ∪ std union, per-instrument tolerance), so the two sides are directly
comparable.

The bracket came out ~0.001 wide against a ~0.49 trend. Ambiguity is a non-issue here; the trend
is fully identified.

## Result

400 mols / 7,252 spectra, nist20 **test**, DAG `nist20_ma_mi_d3_h4_isoFalse_smpFalse`,
run `4ews8j6f` (mean `cos_hun_inst_tol` = 0.7506 — a representative baseline).

| NCE bin | n | true retained | pred retained | pred−true | MAE | **MAE / true** |
|---|---|---|---|---|---|---|
| 0–20 | 827 | 0.8378 | 0.8630 | +0.0252 | 0.0825 | **9.8%** |
| 20–40 | 2396 | 0.7291 | 0.7278 | −0.0013 | 0.0781 | **10.7%** |
| 40–60 | 1313 | 0.5978 | 0.5962 | −0.0016 | 0.0602 | **10.1%** |
| 60–80 | 927 | 0.4960 | 0.4998 | +0.0039 | 0.0493 | **9.9%** |
| 80–100 | 624 | 0.4529 | 0.4588 | +0.0059 | 0.0465 | **10.3%** |
| 100–120 | 495 | 0.4071 | 0.4164 | +0.0093 | 0.0411 | **10.1%** |
| 120–160 | 471 | 0.3863 | 0.3968 | +0.0105 | 0.0397 | **10.3%** |
| 160+ | 199 | 0.3497 | 0.3683 | +0.0186 | 0.0369 | **10.6%** |

Pearson r(nce, retained) = −0.590 (true), −0.591 (max-attribution).

## Findings

**1. The chemistry premise is correct.** True retained fraction falls 0.838 → 0.350 across NCE
(r = −0.59). Mass really does move to smaller fragments as energy rises.

**2. The model already reproduces it.** Predicted falls 0.863 → 0.368. The signed bias never
exceeds 0.025 against a 0.49 trend (~4%), and is under 0.01 through the middle of the range.

**3. The extent error is CE-invariant.** Per-spectrum MAE falls 0.083 → 0.037 in absolute terms,
but that is entirely a scale effect — **relative error is flat at ~10% in every CE bin (9.8%–10.7%)**.

> ⚠️ Correction: an earlier reading of this table claimed the model "gets more accurate at high CE"
> from the absolute MAE decline. That was wrong. The coordinate's own magnitude shrinks by the same
> factor. The correct statement is that extent error is **constant in relative terms across CE**.

## Why this closes the line

The target is the *CE-dependent* collapse: cos 0.93 (NCE 0–20) → 0.64 (NCE 120–200).

Extent error is CE-**invariant** (~10% everywhere). A CE-invariant error cannot explain a
CE-dependent degradation. Fixing extent allocation perfectly would buy a roughly uniform gain at
all energies — it would not close the CE gap, which is the thing we are chasing.

And the flow head's stop probability does not even target the ~10% residual. Its mechanism is
specifically *CE-conditioned* extent shifting — making mass move deeper as energy rises. The model
already does that to within 4%. The mechanism is aimed at the one part of the problem that is
already solved.

Dead as a result, without a GPU run:

- the CFM-flow **stop head** — its only real mechanism
- **`depth_ce_bias`** — no need to sweep it; the additive version of the same already-solved thing
- a **CE × `num_cut_bonds` bias** — the un-contaminated-coordinate version; dies with the rest

## Where the gap actually is

The model puts the right total mass at the right fragmentation extent, at every energy, and then
picks the **wrong fragments at that extent**. Since extent error is CE-flat while cosine is strongly
CE-dependent, the CE-dependent component must live in fragment **identity**.

That is sibling discrimination, which is what **`FragmentCompetitionHead`** does (CE-conditioned
top-K transformer, zero-init delta on the bag-score path) — already built, 18 tests, NaN
root-caused and fixed, config ready, **never GPU-confirmed**. It is now the only live bet on this
axis, and it is correctly aimed: the discrimination problem plausibly worsens with CE because more
fragments share a size as fragmentation proceeds, which is exactly what cross-fragment competition
addresses.

This independently corroborates the [gap decomposition](dag_oracle_ceiling_nist20.md#gap-decomposition-and-overfit-ladder-from-local-memory-verified-2026-10-06)'s "rank/identity, not calibration"
conclusion on a second, unrelated axis.

## Confirming the identity signal is present (cc-corruption probe)

"Wrong fragment at the right extent" has two possible causes: the model cannot *see* which fragment
is which (representation not used / redundant), or it sees them fine but cannot *compare* them. These
demand opposite fixes — more features vs. a competition mechanism — so the branch has to be resolved
before building anything.

The `cc` fragment feature is the atom-membership mask (which atoms of the molecule belong to a
fragment), pooled out of the mol GNN to give each fragment its structure-aware embedding.
`scripts/debug/debug_cc_corruption_eval.py` flips each bit of that mask independently with
probability `frac` at eval time on the frozen 9t63zils baseline — only `cc`, with `boundary_pair`
left on the true mask, no training. nist20 test, ~2,400 spectra, 17-element DAG dir:

| cc corruption frac | cos_hun_inst_tol |
|---|---|
| 0.00 (baseline) | 0.8209 |
| 0.05 | 0.8023 |
| 0.10 | 0.7849 |
| 0.20 | 0.7563 |
| 0.40 | 0.7189 |
| 0.80 | 0.6511 |
| 1.00 (all bits inverted, padding included) | 0.6106 |

Monotonic; full inversion costs **−0.21 cos (−26% relative; magnitude unverified, see below)**, visible even at 5%. So the model
**uses** precise fragment atom identity — this rules out the "coasts on formula/size, ignores
structure" failure mode. The inputs to the fragment-vs-fragment comparison are already good; what is
missing is the comparison itself.

Combined with the extent null above, the CE→identity gap is neither an extent-prior problem nor a
representation problem. It is **comparative scoring**: with `pyg_edges: False`, each fragment is
scored by an independent per-node MLP, and nothing relates same-size siblings except the
global-softmax normalizer. This closes the "more features / richer encoder" branch and green-lights
`FragmentCompetitionHead` as the correctly-aimed bet. Full detail and reproduce steps:
`docs/findings/fragnnet/cc_corruption_probe_finding.md`.

**Caveat (2026-10-12):** the probe's flips also hit padding bits that indexed atoms of other
molecules in the batch, so the magnitudes above are unverified; see the
[probe finding's caveats](cc_corruption_probe_finding.md#caveats).

## Caveats

- **FT only.** Other `inst_type` values did not clear the 50-per-bin threshold. CID and QTOF — the
  known-weak instruments — are unmeasured on this axis. Re-run with `--limit_mols 4000` before
  treating them as closed.
- 400 mols, single run (`4ews8j6f`).
- This measures the *marginal over extent*. It does not rule out a per-spectrum extent error that
  is CE-dependent in some subpopulation, only that the aggregate relative error is not.
- The ~10% relative extent error is real and not nothing — it is simply CE-flat, so it is not the
  CE story. It could still be worth a CE-*unconditioned* intervention.

## Method note for future work

Before building any conditioned module, project both the true and predicted intensity onto the
candidate coordinate, bin by the conditioning variable, and compare **per-spectrum MAE — and its
ratio to the coordinate's own scale**. The signed mean cancels; the absolute error moves with the
coordinate's magnitude. Only the relative error answers "is this axis where the conditioning-dependent
failure lives?"

## Reproduce

```bash
python scripts/eval/ce_extent_headroom.py \
    --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \
    --split test \
    --pred_hdf5 output/inference_eval/4ews8j6f/predictions_test.hdf5 \
    --limit_mols 400 \
    --out_fp /tmp/extent.csv
```

Omit `--pred_hdf5` for the data-only half (does the trend exist at all — needs no model).
