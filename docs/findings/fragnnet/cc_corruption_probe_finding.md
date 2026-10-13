# cc-mask corruption probe: is the fragment identity representation used?

> **Status:** FINDING · **Updated:** 2026-10-12
> Fragment identity representation IS used (cos 0.82 -> 0.61 under mask corruption); the gap is comparative scoring, not features. Closed. **The corruption leaked atoms across molecules (see the first caveat); treat the magnitudes as unverified.**

**Date:** 2026-07-16
**Script:** `scripts/debug/debug_cc_corruption_eval.py` (on `research/ce_perfrag_features`)
**Checkpoint:** `nist20_inchikey_fraggnn_d3_ma_mi_all_9t63zils.ckpt` (baseline, mean cos ~0.75)
**Setup:** nist20 test, 600 batches × bs 4 (~2,400 spectra), DAG dir
`nist20_ma_mi_d3_h4_isoFalse_smpFalse` (17-element, matches the checkpoint).

## Question

The `cc` fragment feature is the atom-membership mask: which atoms of the molecule belong to a
fragment, pooled out of the mol GNN to give each fragment its structure-aware embedding. Is that
precise atom identity actually *used* by the model, or is it redundant with the fragment's formula,
degree, depth, and heavy-atom count? If it is ignored, "the model can't tell same-size fragments
apart" is a representation-not-used problem and no discrimination head can help until it is fixed.

The probe flips each bit of the cc mask independently with probability `frac` at eval time (only
the `cc` feature — `boundary_pair` keeps the true mask), on a frozen checkpoint. No training.

## Result

| cc corruption frac | cos_hun_inst_tol |
|---|---|
| 0.00 (baseline) | 0.8209 |
| 0.05 | 0.8023 |
| 0.10 | 0.7849 |
| 0.20 | 0.7563 |
| 0.40 | 0.7189 |
| 0.80 | 0.6511 |
| 1.00 (all bits inverted, padding included) | 0.6106 |

Monotonic. Inverting the atom-membership mask costs **−0.21 cos (−26% relative; magnitude unverified, see Caveats)**, and even
5% corruption is immediately visible (−0.019).

## Interpretation

**The fragment identity representation is load-bearing, and the model uses it.** This rules out the
"model ignores atom identity, coasts on formula/size" failure mode. Corrupting precise atom
membership degrades predictions smoothly and substantially.

So the CE→identity gap (see `docs/findings/fragnnet/ce_extent_headroom_finding.md`: the model gets fragmentation
*extent* right at all energies but picks the wrong fragment *at* that extent, increasingly so at
high CE) is **not** a case of an unused representation. The atom-identity signal is present and
consumed. The deficit is therefore in how fragments are *scored against each other*, not in whether
the model can see them.

That is the comparative/competition hypothesis, not a feature or representation-plumbing problem:
with `pyg_edges: False`, each fragment is scored by an independent per-node MLP and the only thing
relating fragment A to B is the global-softmax normalizer. "Which of these same-size candidates
actually forms" is inherently comparative, and the architecture has no mechanism to compute it. The
probe says the inputs to that comparison are already good; what's missing is the comparison.

**→ green-lights `FragmentCompetitionHead`** (built, 18 tests, NaN root-caused, config ready on this
branch — `frag_competition` hparams present, never GPU-confirmed) as the correctly-aimed next bet,
and argues *against* spending effort on more fragment features or a richer per-fragment encoder.

## Caveats

- **Cross-molecule contamination (found 2026-10-12).** The flips covered all 128 mask bits,
  including padding beyond each molecule's atom count, and the model checked bounds only
  against the batch total. Flipped padding bits of molecules 1–3 in each batch of 4 pulled in
  atoms of later molecules; those of the last molecule were dropped. NIST20 test-split
  spectra have a median of 19 heavy atoms (p10 11, p90 34), so about 83% of the 128 bits are
  padding and most flips at every `frac` were padding (whole-split statistic; the probe's
  ~2,400-spectrum subset was not measured separately). The table therefore
  measures added foreign atoms as much as lost membership information, and the `1.00` row is
  a deterministic inversion (the complement plus foreign atoms), not a random mask. A monotone
  drop is still evidence the cc pooling is read, but the −0.21 magnitude and the
  dose-response shape are unverified. The leak is fixed in the
  [simplification audit](fragnnet_simplification_audit_2026_10_12.md#1-cc-mask-padding-bits-leaked-across-molecules-fixed-in-the-working-tree);
  rerunning the probe on the fixed code would give the clean curve. Not rerun: the script
  exists only on `research/ce_perfrag_features`, whose model still has the leak, so a rerun
  needs the script copied to the fixed branch and its `fragnnet.utils.script_utils` import
  updated to `fragnnet.inference.runners`.
- Baseline here is 0.82 on this ~2,400-spectrum FT-heavy subset, higher than the full-test 0.75;
  the *shape* (monotonic, large) is the finding, not the absolute level.
- FT-dominated subset; not stratified by CE. A sharper version would measure whether the corruption
  sensitivity itself grows with NCE (i.e. is identity *more* load-bearing at high CE) — that would
  directly tie the probe to the CE gap. Not yet done.
- 8 GB card forced bs 4 + expandable_segments; some giant-DAG molecules still spike memory.

## Reproduce

```bash
cd /home/feiw/gitRepos/frag-gnn-perfrag   # worktree on research/ce_perfrag_features
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$PWD/src python scripts/debug/debug_cc_corruption_eval.py \
  --custom_fp config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_train_metrics.yml \
  --model_ckpt saved_ckpts/nist20_inchikey_fraggnn_d3_ma_mi_all_9t63zils.ckpt \
  --eval_split test --eval_batch_size 4 --batch_cutoff 600 \
  --auxiliary_scores cos_hun_inst_tol \
  --corruption_fracs 0.0 0.05 0.1 0.2 0.4 0.8 1.0
```

Note: `--auxiliary_scores cos_hun_inst_tol` (unsuffixed); `cos_sim` becomes `cos_sim_0.01` and errors
if requested raw. Requires the 17-element DAG dir; the checkpoint's frag embedder is 936-wide.
