# Connected-fragment beam: reuse and objective equivalence

> **Status:** FINDING · **Updated:** 2026-09-17 · Source and algebra review, plus a local
> candidate-only measurement of the predicted size and topology prior. No training and no code
> changes. Speed numbers are quoted from the existing
> [beam speed audit](connected_fragment_beam_speed_2026_09_17.md) and the
> [FastGNN proposal plan](../../plans/masster/masster_fast_gnn_proposal.md).

## Result

Measured on trained bond logits at a matched unique-fragment budget, feeding both selectors
identical logits so only the selection objective differs.

1. **The beam cannot reach most of the probability mass, at any budget.** Captured mass is flat
   at **0.8695** from K = 256 to 1,024 against **1.0000** for `cut_masks`. It is complete below
   half the molecule (reach 1.000 and 0.999 by count) and structurally limited above it, reaching
   0.174 of fragments above 0.75 of the molecule — the band holding 42.9% of all mass. Growth
   stops at `ceil(n / 2)`, so anything above half arrives only as a connected mirror, and a large
   fragment needing several cuts leaves a disconnected remainder. That is the
   precursor-minus-two-neutral-losses shape, missing by construction. **Architectural, not a
   scoring artifact.**

2. **Its score carries a size and topology prior that is chemically backwards.** `beam(f) =
   log P(m = ∂f) − I_f̄`, and the dropped term makes the score fall monotonically with fragment
   size. The beam keeps the smaller side of a cut about 3:1 to 5:1 and implies "charge stays on
   the smaller fragment". The boundary-cut cap and the candidate budget are excluded as causes;
   more budget makes it worse.

3. **That prior is entirely the scoring change.** The original prototype has the same
   architecture but ranks by mask log probability, and emits a near-symmetric distribution:
   0.485 above half against the installed beam's 0.174 and `cut_masks`' 0.500.

4. **But mask scoring alone does not rescue it.** The prototype captures *less* mass, 0.757
   against 0.869, because a fragment and its complement then score identically and it spends two
   budget slots per mask. The two beam variants fail in complementary ways and neither approaches
   `cut_masks`. Escaping both requires emitting every component of one mask, which is exactly
   `cut_masks`' assembly.

5. **Nine of roughly thirteen beam mechanisms already exist in MASSter**, including the
   union-find components that would replace the dense-reachability complement pass responsible
   for 447 of the 493 ms regression. That replacement is worth doing if the beam stays in the
   tree as a diagnostic, but it does not change conclusions 1 through 4.

6. **The two audit fixes barely change the output.** Pre-fix and post-fix emit identical fragment
   sets in 18 of 24 configurations and differ by one or two fragments in 4,001 otherwise. The
   447 ms bought determinism and a rare ranking correction, not materially different candidates.

Taken together: the beam's objective nearly coincides with `cut_masks` up to a charge prior, so
it has to win on search cost or candidate coverage, and it does neither. Reproduction commands
are below; all measurements are candidate construction only, with the limits recorded at the end.

## Objective equivalence

Notation for a fragment atom set `f` with boundary `∂f`, internal bonds `int(f)`, and bonds
with both endpoints outside `f` written `int(f̄)`:

```text
B    = Σ_{e ∈ ∂f}     log p_e
I_f  = Σ_{e ∈ int(f)} log(1 − p_e)
I_f̄  = Σ_{e ∈ int(f̄)} log(1 − p_e)
```

Every bond is in exactly one of these three sets, so the mask that cuts precisely `f`'s
boundary has `log P(m = ∂f) = B + I_f + I_f̄`, and

```text
beam(f)   = B + I_f            = log P(m = ∂f) − I_f̄
cut_masks = B + I_f + I_f̄ + log P(charge = f | m)   (+ mechanism term)
```

### The charge head is replaced by a fixed prior

The two agree iff `log P(charge = f | m) = −I_f̄ + c`. Since the charge term must normalize
over the mask's components, that identification is **exact for the relative weighting of
components within one mask**, and leaves a per-mask offset `log Z_m` when fragments from
different masks are compared. `Z_m` is not constant per molecule, so it does not cancel in the
per-molecule softmax.

Read plainly: the beam breaks the fragment/complement symmetry with `I_f − I_f̄`, a fixed
size-and-fragility prior — charge goes to the side whose complement is less likely to survive
intact — where `cut_masks` breaks it with a learned head. That is the substantive modelling
difference between the two, and it is a narrower one than "different objective".

### The dropped term is a size and topology prior

Because `I_f = Σ_{int(f)} log(1 − p_e) ≤ 0` and grows more negative with every internal bond,
`B + I_f` charges a fragment for its own bonds but not for the rest of the molecule. At fixed
boundary the score therefore decreases monotonically with fragment size, and a single atom,
having no internal bonds, attains `I_f = 0`. The formulation systematically favours:

1. **Small fragments.** Singletons are the maximum of the size term.
2. **Acyclic fragments.** A connected `n`-atom fragment has at least `n − 1` internal bonds,
   with equality only for a tree, so ring systems pay extra. The ring-reconnection gap below
   undercounts the same fragments, so rings are penalized twice by independent mechanisms.
3. **Stable interiors with fragile boundaries** — the one component of the prior that is
   chemically reasonable.

`cut_masks`' `B + I_f + I_f̄` carries neither bias 1 nor 2: every bond is counted exactly once
per mask, so a fragment and its complement score identically. **The size and topology prior is
exactly what dropping `I_f̄` introduces.**

Order of magnitude, for a tree fragment's `n − 1` internal bonds at a uniform non-breaking `p`:

| typical `p` | 5-atom | 20-atom | ratio |
| --- | ---: | ---: | ---: |
| 0.1 | −0.42 | −2.02 | ~5× |
| 0.3 | −1.43 | −6.78 | ~220× |

At the operating point implied by `mask_cut_fraction_4plus = 0.77`, the structural preference
for small fragments is plausibly a factor of hundreds before any learning. These are worked
values at assumed `p`, not measurements. The measured section below confirms the direction and
that it strengthens with `p`, but does not verify these particular ratios.

The optimizer cannot simply absorb this. A single bond's logit enters with different roles in
different fragments — internal to some, boundary to others, outside for the rest — so per-bond
parameters cannot independently set a size prior. It is an inductive bias, not a
reparameterization.

Chemically, since `score(f) − score(f̄) = I_f − I_f̄`, the smaller and more stable side wins: the
beam's implicit charge model is "charge stays on the smaller fragment". That is the quantity
`cut_masks` spends a learned charge head on, and it runs against MS/MS practice, where charge
retention tracks proton affinity and basic sites rather than size, and where observed fragments
are typically large pieces remaining after a small neutral loss.

### The ring-reconnection gap

The beam scores the event "boundary all cut **and** all internal intact". The component event
is weaker: an internal bond in a cycle may break while `f` stays connected through the other
arc. Summing over internal cut subsets `S` that leave `f` connected,

```text
P(f is a component) / exp(beam(f))  =  Σ_{S : f−S connected} Π_{e ∈ S} p_e/(1−p_e)
                                    ≈  1 + Σ_{e ∈ cycle bonds of f} p_e/(1−p_e)
```

- **Induced-acyclic `f`:** every internal bond is a bridge, the sum is empty, and the beam's
  closed form is the **exact** marginal — while the `cut_masks` top-K sum is a truncated lower
  bound converging to the same value. On tree fragments the beam is the better estimator.
- **Ring-containing `f`:** the beam strictly undercounts, by a factor that depends entirely on
  the trained ring-bond `p_e`. At `p = 0.02` a six-bond ring gives ×1.12 (≈0.11 nats,
  negligible after normalization). At `p = 0.3` it gives ×3.6 (≈1.3 nats, not negligible).

This is a systematic tilt toward ring-poor fragments, so it partially cancels between
candidates of similar ring content. Whether it matters is an empirical question about where
the trained model puts ring-bond probability, not a modelling assumption.

### Search exactness runs the other way

`cut_masks` selection is exact top-K under its own objective (heap enumeration over subsets).
The beam is approximate under its own: finite beam width, dedup underfill, connected
complements only. The plan's
[connected-fragment audit](../../plans/masster/masster_fast_gnn_proposal.md#connected-fragment-audit-2026-09-17)
records two graphs where a higher-scoring reachable fragment was lost. So the beam trades an
exact search over masks for an approximate search over fragments.

### Closing the question

1. **Partly done.** The overall trained `p_e` distribution is measured below (mean 0.096,
   median 0.0007), which places the model firmly in the small-`p` regime and makes the
   ring-reconnection correction small for most bonds. The **ring-bond-restricted** histogram and
   the per-fragment `Σ_{e ∈ cycle bonds} p_e/(1−p_e)` remain unrun, so the ring gap's exact size
   is still unquantified. Note this cuts against the earlier `mask_cut_fraction_4plus = 0.77`
   expectation, which came from a different model.
2. **Unrun.** Compare the beam's implicit prior `exp(−I_f̄)` against a trained charge head's
   actual component split on the same masks.
3. ~~Compare the emitted fragment **size and ring-count distributions** between the two
   proposal methods at a matched fragment target.~~ **Done 2026-09-17** — see the measured
   sections below, on synthetic and trained logits, with both confounds excluded and the
   scoring change isolated against the original prototype.

## Measured: emitted fragment distributions (2026-09-17)

The predicted size and topology prior is confirmed. Identical bond logits were fed to both
selectors at a matched unique-fragment target, isolating the selection objective.

At `p` mean 0.155 (logits `N(-2, 1)`), the realistic regime for a model that keeps most bonds:

| Selector | fragments | size mean | size median | singleton | ≤3 atoms | acyclic | mean fraction of molecule | fraction > half | fraction ≥ 0.75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `connected_fragment_beam` | 7,289 | 6.63 | 6 | 0.083 | 0.246 | **0.801** | **0.346** | **0.167** | **0.079** |
| `cut_masks` | 7,904 | 11.33 | 9 | 0.057 | 0.144 | 0.550 | 0.558 | 0.590 | 0.319 |

`cut_masks` components come from masks whose components partition the molecule, so both sides
of a cut are naturally present and a near-symmetric 0.59 above the half line is expected by
construction. The beam must generate complements explicitly, and **retains the smaller side
about 5:1**. It proposes **roughly four times fewer large fragments** — exactly the
precursor-minus-small-neutral-loss species that dominate real MS/MS — and **1.5x more acyclic
fragments**, which is the topology half of the prior.

### The bias strengthens as bonds become more breakable

Fraction of emitted fragments larger than half the molecule:

| `p` mean | beam | `cut_masks` |
| ---: | ---: | ---: |
| 0.069 | 0.212 | 0.580 |
| 0.155 | 0.167 | 0.590 |
| 0.330 | 0.101 | 0.499 |
| 0.506 | 0.085 | 0.329 |

This is the predicted direction: each internal bond costs `log(1 − p)`, which grows more
negative as `p` rises. The beam's mean fragment size is also comparatively insensitive to the
logits — 7.68 → 5.13 across a 7x change in mean `p`, against 11.34 → 7.12 for `cut_masks` —
consistent with its output being driven by a structural prior rather than by the learned bond
probabilities.

### Two controls exclude the obvious confounds

**Not the boundary-cut cap.** Raising `max_boundary_cuts` leaves the distribution frozen once
it stops binding, so the bias is the score and not the cut budget:

| `max_boundary_cuts` | fragments | size mean | > half | ≥ 0.75 |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 5,686 | 7.19 | 0.196 | 0.118 |
| 5 | 7,289 | 6.63 | 0.167 | 0.079 |
| 8 | 7,317 | 6.63 | 0.169 | 0.079 |
| 12 | 7,317 | 6.63 | 0.169 | 0.079 |
| 20 | 7,317 | 6.63 | 0.169 | 0.079 |

**Not the candidate budget — it moves the wrong way.** Extending the fragment target buys more
*small* fragments, because the ranking is ordered by size and a longer list descends further
into it:

| target = beam width | fragments | size mean | > half | ≥ 0.75 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 7,317 | 6.63 | 0.169 | 0.079 |
| 512 | 12,649 | 7.44 | 0.142 | 0.054 |
| 1,024 | 20,004 | 8.38 | 0.139 | 0.039 |

### Generation versus ranking: a separate coverage ceiling

The complement is generated, so the large-fragment deficit is not purely a ranking effect.
Pushing the budget toward exhaustion on a 16-molecule batch (atoms 13 / 20 / 95) separates the
two:

| target = beam width | fragments | mean size | count > half | count ≥ 0.75 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 4,001 | 6.71 | 662 | 305 |
| 512 | 7,333 | 7.69 | 1,165 | 453 |
| 1,024 | 12,815 | 8.54 | 1,810 | 548 |
| 2,048 | 21,412 | 9.45 | 2,286 | 574 |
| 4,096 | 37,796 | 10.40 | 3,461 | 595 |

Total candidates grow 9.4x while fragments at or above 0.75 of the molecule **saturate near
595**: the last step adds 16,384 candidates and 21 large ones. That is a coverage ceiling, not
a budget limit. Two distinct effects follow:

1. **Ranking.** At the production target of 256 the beam keeps 305 of roughly 595 reachable
   large fragments, about 51%. This is the size prior, and the scoring change addresses it.
2. **Coverage.** Growth stops at `ceil(n / 2)`, so every fragment above half arrives only as a
   mirror, and mirrors are retained only when connected. A connected fragment larger than half
   **whose complement is disconnected is unreachable by either route.** The plan records this as
   the disconnected-complement limitation; the saturation is its external signature. The shape
   is chemically common: a precursor losing two small neutrals from different positions leaves
   one large connected fragment and a disconnected remainder.

The scoring change closes both, because `log P(∂f) = C_mol + Σ_{∂f} logit` needs only the
boundary — no complement and no internal bonds — so a fragment can be scored from its cut set
whether or not the remainder is connected. Paired with union-find, which returns every component
of a cut set and not just two, the coverage hole closes with the same edit that removes the
ranking bias and the complement pass. This also bears on gate 1 in the plan: emitting all
components of one cut set is a different question from enumerating several cut sets per atom
set.

The saturation count is not a coverage comparison against `cut_masks`; the two runs used
different batches. A matched-batch reachable-set comparison remains open.

### Coverage per unit budget, and the architectural mass ceiling

The remaining argument for the beam was enumeration efficiency: spending budget directly in
fragment space rather than on masks that duplicate fragments. It does not hold. Every fragment
in the reference universe, the union of both selectors at large budget, was scored by the
objective the proposal would adopt, `log P(m = ∂f) = C_mol + Σ_{∂f} logit`, and each selector
measured by captured probability mass at matched budget:

| K | beam mass | `cut_masks` mass | beam ms | `cut_masks` ms | beam ms/fragment | `cut_masks` ms/fragment |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.661 | 0.855 | 1,480 | 34 | 0.370 | 0.0083 |
| 512 | 0.685 | 0.921 | 2,799 | 89 | 0.382 | 0.0112 |
| 1,024 | 0.695 | 0.976 | 4,853 | 205 | 0.379 | 0.0141 |
| 2,048 | 0.701 | 0.983 | 9,408 | 500 | 0.439 | 0.0203 |

**The beam's captured mass saturates near 0.70** across an eightfold budget increase, while
`cut_masks` reaches 0.983. `cut_masks` per-fragment cost does degrade with budget, 0.0083 to
0.0203 ms, but stays roughly twentyfold below the beam. Both selectors ran on CPU here, which is
the beam's unfavourable device; the mass ceiling, however, is device independent.

Decomposing the reference universe by fragment size locates that ceiling exactly:

| fragment size, fraction of molecule | reference fragments | share of total mass | beam reaches, by count | mass beam reaches |
| --- | ---: | ---: | ---: | ---: |
| [0, 0.25) | 7,778 | 0.219 | 1.000 | 0.219 |
| [0.25, 0.50) | 22,195 | 0.137 | 0.999 | 0.136 |
| [0.50, 0.75) | 11,226 | 0.172 | 0.647 | 0.082 |
| [0.75, 1.0] | 5,477 | **0.472** | **0.109** | 0.270 |

The beam is **complete below half the molecule and structurally blind above it**, and above half
is where the mass is: the largest band alone holds 47.2% of total probability mass and the beam
reaches 10.9% of its fragments. The captured total, 0.707, matches the measured 0.701 saturation,
so the ceiling is fully accounted for.

The mechanism is visible in the top band, where the beam reaches only 10.9% of fragments but
captures 57% of that band's mass: a large fragment produced by a **single** bond cut always has a
connected complement and is therefore reachable, while a large fragment needing several cuts
generally leaves a disconnected remainder and is not. That is the precursor-minus-two-small-
neutral-losses shape, and it is missing by construction.

This is a coverage defect **independent of the scoring**. Mask scoring corrects the ranking bias
but does not make these fragments reachable; only emitting every component of a cut set does,
which is `cut_masks`' assembly. The enumeration-efficiency argument therefore fails on its own
terms, and it was the last one supporting the beam as a distinct proposal method.

### Confirmed on trained bond logits

The measurements above use synthetic logits. Repeating the coverage scan with **trained** bond
logits captured from the MASSter checkpoint `zlv0bli8` (epoch 41) confirms the direction and
the mechanism, and **revises the magnitude upward in the beam's favour**. Only the encoder,
bond-break head and charge head are loaded, which is the whole proposal path; the checkpoint's
intensity head predates the current architecture and is never used.

The trained model is far more confident than the synthetic sweep assumed:
`p(cut)` mean 0.096, p10 0.0000, p25 0.0000, **p50 0.0007**, p75 0.040, p90 0.386. Most bonds
essentially never break.

| selector | K | size mean | acyclic | > half | ≥ 0.75 | **captured mass** | ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `connected_fragment_beam` | 256 | 7.48 | 0.743 | 0.174 | 0.087 | **0.8694** | 220.9 |
| `cut_masks` | 256 | 10.12 | 0.571 | 0.500 | 0.262 | **0.9997** | 34.7 |
| `connected_fragment_beam` | 1,024 | 8.68 | 0.764 | 0.148 | 0.045 | **0.8695** | 455.7 |
| `cut_masks` | 1,024 | 11.96 | 0.546 | 0.586 | 0.287 | **1.0000** | 205.8 |

**The ceiling is 0.87 on trained logits, not the 0.70 measured synthetically** — but it is still a
ceiling: 0.8694 to 0.8695 across a fourfold budget increase is flat to four decimal places, while
`cut_masks` is essentially complete at K = 256. The beam remains about six times more expensive
per fragment on this batch.

| fragment size, fraction of molecule | reference fragments | share of total mass | beam reaches, by count | mass beam reaches |
| --- | ---: | ---: | ---: | ---: |
| [0, 0.25) | 2,807 | 0.194 | 1.000 | 0.194 |
| [0.25, 0.50) | 15,120 | 0.217 | 1.000 | 0.217 |
| [0.50, 0.75) | 7,460 | 0.160 | 0.685 | 0.096 |
| [0.75, 1.0] | 2,788 | **0.429** | **0.174** | 0.363 |

The structure is unchanged: complete below half the molecule, sharply limited above it, with the
top band holding the largest share of mass. The beam does better here than under synthetic logits
for an understandable reason — a confident model concentrates mass on **single-cut** masks, whose
large fragment always has a connected complement and is therefore reachable. What stays missing
is the multi-cut large fragments, which is precisely the predicted failure mode.

So the size and topology prior, the above-half coverage limit and the flat ceiling all reproduce
on real trained weights. The cost of the limit is about 13% of probability mass rather than 30%.

### The original prototype isolates the scoring change

The standalone research prototype `others/fast_gnn.py` shares the beam's architecture — growth to
`ceil(n / 2)`, connected mirrors, the same budget — but ranks by the **mask log probability**
rather than by `B + I_f`. Running it on the same trained logits therefore isolates the scoring
decision from everything else.

| selector | K | fragments | size mean | > half | captured mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| repo beam, `B + I_f` | 256 | 3,523 | 7.48 | **0.174** | **0.8694** |
| prototype, `log P(∂f)` | 256 | 3,801 | 9.85 | **0.485** | **0.7569** |
| `cut_masks` | 256 | 3,691 | 10.12 | 0.500 | **0.9996** |
| repo beam, `B + I_f` | 1,024 | 9,608 | 8.68 | 0.148 | 0.8694 |
| prototype, `log P(∂f)` | 1,024 | 11,843 | 11.36 | 0.560 | 0.7570 |
| `cut_masks` | 1,024 | 9,304 | 11.96 | 0.586 | 1.0000 |

**The size and topology prior is entirely the scoring change.** With mask scoring the prototype
emits a nearly symmetric distribution, 0.485 above half against `cut_masks`' 0.500, versus 0.174
for the installed beam. Architecture, growth depth and mirroring are identical, so nothing else
can account for it. The two beams agree on only about half their fragments, Jaccard 0.49 at
K = 256.

**But mask scoring costs budget efficiency.** The prototype captures *less* probability mass,
0.757 against 0.869, despite the better size distribution. The likely explanation, inferred from
the counts rather than proven: under mask scoring a fragment and its complement score
identically, so the prototype spends two budget slots per mask and covers roughly half as many
distinct masks, while the installed beam's size prior keeps mostly one side and is accidentally
budget-efficient.

The two beam variants therefore fail in complementary ways and **neither approaches
`cut_masks`**. Correcting the size bias halves the effective budget; keeping the budget
efficiency keeps the bias. Escaping both requires emitting every component of a mask from one
enumeration slot, which is exactly `cut_masks`' assembly. This is the same conclusion the algebra
reaches, measured from an independent direction.

### Pre-fix versus post-fix: the correctness fixes barely change the output

Every measurement above uses the **post-fix** beam: `masster_fast_gnn.py` is unchanged
between `dca3d0f9` and the current tree, so the ceiling is the corrected implementation's, not an
artifact of the audited defects. Comparing the two revisions' emitted fragment sets directly, on
a fixed batch across three logit seeds, two logit means, two budgets and two boundary-cut caps:

| configurations | identical emitted set | differing |
| --- | ---: | ---: |
| 24 | 18 | 6 |

Where they differ, the difference is **one or two fragments out of 4,001**, about 0.03%, and at
the smaller budget of 64 the two agree everywhere. So the two audit fixes cost **447 of 493 ms**
and alter roughly three candidates in ten thousand on real molecular graphs.

That is not an argument that the fixes were wrong. The premature-pruning counterexample is
genuine, and the stable-tie ordering exists to make candidate selection **invariant to batch
composition** — a property this comparison cannot see, because it holds the batch fixed. The
useful conclusion is narrower: the expensive per-round complement pass buys determinism and a
rare ranking correction, not materially different candidates, while the defect that actually
limits the method — the coverage ceiling above half the molecule — is untouched by either
revision.

## Reproducing this analysis

All measurements come from
[scripts/debug/analyze_proposal_fragment_coverage.py](../../../scripts/debug/analyze_proposal_fragment_coverage.py),
run from the repository root with the configured `FRAGNNET-GPU` interpreter. Candidate
construction only: no checkpoint, training or spectra are involved.

```bash
FRAGNNET_PYTHON=/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python

# Emitted size, ring and complement-asymmetry distributions across the logit sweep,
# and the boundary-cut / budget controls. 32-molecule batch.
"$FRAGNNET_PYTHON" scripts/debug/analyze_proposal_fragment_coverage.py \
  --mode distributions --batch-size 32 --subsample-size 256 --output /tmp/beam_distributions.json
"$FRAGNNET_PYTHON" scripts/debug/analyze_proposal_fragment_coverage.py \
  --mode controls --batch-size 32 --subsample-size 256 --output /tmp/beam_controls.json

# Captured mask-probability mass, top-K recall and the size-band decomposition.
# 16-molecule batch; add --coverage-budgets 256 512 1024 2048 for the saturation table.
"$FRAGNNET_PYTHON" scripts/debug/analyze_proposal_fragment_coverage.py \
  --mode coverage --batch-size 16 --subsample-size 128 --output /tmp/beam_coverage.json

# Pre-fix versus post-fix emitted sets. Extract the historical revision first.
git show dca3d0f9:src/fragnnet/frag/masster_fast_gnn.py > /tmp/prefix_beam.py
"$FRAGNNET_PYTHON" scripts/debug/analyze_proposal_fragment_coverage.py \
  --mode implementations --beam-implementation /tmp/prefix_beam.py \
  --output /tmp/beam_implementations.json

# The same coverage scan on TRAINED bond logits, which needs no matching intensity head.
"$FRAGNNET_PYTHON" scripts/debug/analyze_proposal_fragment_coverage.py \
  --mode coverage --checkpoint wandb/run-20260815_150218-zlv0bli8/files/ckpt/last.ckpt \
  --batch-size 16 --subsample-size 128 --coverage-budgets 256 1024 \
  --output /tmp/beam_coverage_trained.json

# The installed beam against the original prototype, which ranks by mask log probability.
# The prototype imports packages absent here; the loader stubs them, since its search is
# pure tensor code. Drop --checkpoint to use synthetic logits instead.
"$FRAGNNET_PYTHON" scripts/debug/analyze_proposal_fragment_coverage.py \
  --mode implementations --beam-implementation others/fast_gnn.py \
  --checkpoint wandb/run-20260815_150218-zlv0bli8/files/ckpt/last.ckpt \
  --batch-size 16 --subsample-size 128 --output /tmp/beam_vs_prototype.json
```

Each run writes a JSON report and logs the tables above. `--logit-seed` selects the synthetic
logit draw; the recorded numbers use the default 17. The distribution and control tables use a
32-molecule batch and the coverage tables a 16-molecule batch, so their fragment counts are not
directly comparable across sections.

### Provenance and limits

- One batch of 32 molecules from the train split of the
  [unmerged CMF beam arm config](../../../config/masster/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast_gnn_proposal.yml),
  in-memory subsample 256, seed 397623. Atom counts 10 / 17 / 40 (min / median / max), 669
  bonds. CPU, `FRAGNNET-GPU` interpreter.
- Beam: `max_fragments = beam_width = 256`, `max_boundary_cuts = 5` unless varied.
  `cut_masks`: `max_masks = max_unique_components = 256`, `max_enumerated_masks = 4096`,
  `include_single_bond_cuts = True`, `candidate_backend = "cpu"`.
- **Budgets are matched on unique fragments, not masks.** `cut_masks` received
  `max_masks = max_unique_components = K` with `max_enumerated_masks = 16K`, the production
  ratio (256 / 256 / 4096), so it may enumerate 16K masks but is measured on K distinct atom
  sets; the beam received `max_fragments = beam_width = K`. Both caps include the intact
  precursor. Matching on output favours the beam: since `cut_masks` is about twenty times
  cheaper per fragment, matching on work instead would have widened the gap.
- **Complement generation was active in every beam run**; `build_fast_gnn_candidates` has no
  option to disable it. Growth stops at `ceil(n / 2)`, so every emitted fragment above half the
  molecule is mirror-derived, and the reported above-half coverage is therefore the ceiling with
  mirroring on rather than without it.
- Fragments are the unique components from `MassterModel._unique_component_memberships`, so
  both selectors are compared on the same emitted-fragment definition. Cycle rank is
  `internal_bonds − atoms + 1`; the `≥ 0.75` column includes the intact precursor, the
  `[0.75, 1)` histogram bin does not.
- **Two logit sources were used.** The distribution, control and size-band tables use synthetic
  logits drawn from a normal distribution with seed 17, so the sweep tests whether *any* learned
  logits could remove the effect. The coverage scan was additionally repeated with **trained**
  logits from checkpoint `zlv0bli8` (epoch 41), loaded non-strictly so that only the encoder,
  bond-break head and charge head must match; the checkpoint's intensity head predates the
  current architecture and is never used. Trained logits reproduce every direction and raise the
  beam's mass ceiling from 0.70 to 0.87. One checkpoint, one epoch, one batch.
- One batch, one molecule-size distribution, candidate construction only. No training, no
  spectral accuracy and no timing conclusions are drawn here.
- Measurements are reproduced by
  [scripts/debug/analyze_proposal_fragment_coverage.py](../../../scripts/debug/analyze_proposal_fragment_coverage.py);
  see the reproduction commands above. The script loads one batch through
  `init_dataset`/`init_dataloader`, generates logits, calls `build_fast_gnn_candidates` and
  `build_masster_candidates` on the same tensors, and summarizes
  `_unique_component_memberships` output.

## Mechanisms that already exist in MASSter

Reviewed [masster_fast_gnn.py](../../../src/fragnnet/frag/masster_fast_gnn.py) against
the existing candidate machinery. Nine of about thirteen distinct mechanisms have an existing
counterpart:

| Beam mechanism | Existing part |
| --- | --- |
| `_connected_complements`: dense `(mols, ≤256, atoms)` reachability plus up to `max_atoms` `bmm` steps | Union-find connected components: `connected_component_atom_masks_many` and `build_component_arrays_raw` in [masster_core.pyx](../../../src/fragnnet/frag/masster_core.pyx); the `connected_components` kernel via [`build_cuda_topology_device_masks`](../../../src/fragnnet/frag/masster_cuda_topology.py) |
| Final boundary/internal membership assembly | `build_component_arrays_raw` / `assemble_tile` already emit `atom_indices`, `atom_component_indices`, `boundary_bond_indices`, `boundary_component_indices` in the adapter's layout |
| `_unique_rows_lex`, three call sites, `C` stable argsorts each | Hash-based atom-set dedup inside `build_component_arrays_raw`; the `deduplicate_tile` CUDA kernel; CPU `_compare_bitsets` + `_merge_sort_indices` |
| 63-bit word packing and unpacking | The codebase's 64-bit bitset convention plus `unpack_component_memberships_raw`. The 63-versus-64-bit split is gratuitous divergence |
| `_group_rank` and its keep masks | `top_mixture_bond_sets_raw` stable rank; the `rank_masks` CUDA kernel; the fragment-cutoff compaction pattern in [masster_model.py](../../../src/fragnnet/model/masster_model.py) |
| Live differentiable rescoring by `index_add` of `logsigmoid` | [`_score_mask_log_probabilities`](../../../src/fragnnet/frag/masster_candidates.py), which also carries mixture support |
| Input validation block | Duplicates the validation in `build_masster_candidates` and the contiguity/uniqueness checks in `_prepare_graph_batch` |
| `mol_atom_count` / `mol_atom_offset` rebuilt by `bincount` | `atom_ptr` is the same thing, and the caller already computes `counts` from it |
| Bond-to-molecule `argsort` | PyG batches already group bonds by molecule; the other builders use per-molecule bond lists instead of re-sorting |

**Genuinely new, and worth keeping:** score-guided connected-set growth with the incremental
boundary update, the beam frontier itself, per-molecule growth depth, and applying the
boundary-cut budget to emitted fragments only. Nothing else in the repository ranks *connected
atom sets* by fragment score; existing selection ranks *cut masks* by mask probability.

One caveat on the reuse list: the internal-bond index arrays exist only because of the
"outside unrestricted" score. Under the mask-probability formulation they disappear entirely,
since `_score_mask_log_probabilities` needs the boundary alone. That is the objective change
discussed above, not a free refactor.

## Existing features the beam currently switches off

The construction guard in `MassterModel.__init__` requires unique mode, one mixture, SMARTS
off, `share_proposal_across_ev` off and a zero mask-probability floor. That disables five
mechanisms that already exist and work for `cut_masks`:

1. **Shared proposal across CE views** (`candidate_group_ids`, representative logic in
   `_prepare_graph_batch`). The largest throughput item: unmerged data carries roughly 18.5
   spectra per molecule, so the search currently repeats per CE view.
2. **Forced prefix** (`include_single_bond_cuts`, `_merge_forced_masks`). The beam hand-rolls
   the intact-precursor reservation and has **no guarantee that single-bond-cut fragments are
   proposed at all** — they must be found by search. A forced prefix is a cheap, deterministic
   coverage floor.
3. **Relative probability floor** (`min_mask_relative_probability`).
4. **SMARTS mechanism rows** — the adapter already carries `mechanism_rule_indices`, and the
   beam writes zeros into it.
5. **Bernoulli-mixture proposals.**

Not guarded but unused: `MassterStaticGraph` (the beam rebuilds adjacency every step) and
`candidate_backend` / `mask_selection_threads`, so the CUDA selection and topology backends are
unreachable from this proposal method.

Separately, the beam's timing record fills six of ten fields with `0.0` and sets
`output_transfer_ms` equal to `probability_scoring_ms`, so beam-versus-`cut_masks` timing rows
are not like-for-like.

## Budget efficiency: `cut_masks` does the same work, in mask space

`cut_masks` reaches the same fragments, but spends its budget on **masks** rather than
fragments, and a fragment `f` is produced by every mask whose cut set contains `∂f` and leaves
`f` connected — exponentially many. The repository already prices this in: the unmerged arm
sets `masster_ranked_mask_budget: 4096` against `masster_unique_fragment_target: 256`,
a **16× enumeration slack** whose only purpose is to reach 256 distinct atom sets. The template
defaults to 256/256, so the 4096 is a deliberate admission that mask-space enumeration is lossy
in fragment space.

The size of that loss is **already logged on every existing run** — no new instrumentation is
needed to settle the efficiency question:

| Metric | What it answers |
| --- | --- |
| `unique_components_per_mask` | Unique fragments obtained per selected mask — the direct efficiency ratio |
| `duplicate_component_pathway_fraction` | `1 − unique / pathways`; the share of component work that is duplicate |
| `latent_pathways_per_unique_fragment` | How many mask pathways collapse into one fragment |
| `productive_mask_fraction` | Share of masks that contributed anything at all |
| `unique_fragment_target_fill_fraction` / `_reached` | Whether the 256 target was met within the 4096 budget |

Two corrections to a purely "beam is leaner" reading:

1. **A mask is not one fragment.** One union-find pass yields *all* of that mask's components at
   once (`components_per_mask`), so mask-space cost per fragment is lower than the 16× slack
   suggests. The realized number is logged.
2. **The beam examines a comparable number of states.** Beam width 256 over `ceil(atoms/2)`
   rounds is roughly 2,600 states at 20 atoms and 9,700 at 76, plus a complement and
   connectivity check per recorded candidate. Its advantage is not fewer states but that each
   state *is* a fragment; its disadvantage is per-state cost.

**The two threads are the same thread.** The duplicate pathways that make `cut_masks` look
wasteful are exactly the terms the beam's closed form replaces: `latent_pathways_per_unique_fragment`
is a count of the pathways that the beam either sums analytically (acyclic fragments, where its
closed form is exact) or drops (ring reconnection, charge alternatives). So `cut_masks`'
redundancy is partly the price of the richer marginal, not pure overhead.

That makes the comparison answerable from artifacts that already exist: read
`unique_components_per_mask`, `duplicate_component_pathway_fraction` and candidate timing from
an existing `cut_masks` run, against `fast_gnn_fragment_count` and `fast_gnn_search_batch_ms`
from a beam run, at matched fragment target. No new benchmark is required for a first answer.

## Prototype comparison: the scoring change created the cost

The workspace prototype `others/fast_gnn.py` (untracked) carries **two scores that disagree**, and the
repo's divergence is how that inconsistency was resolved.

Its *search* accumulates `C_mol = Σ_all log(1 − p_e)` from `−softplus(logit)` and grows by
`deg_logit − 2·to_F`, whose running value is `Σ_{e ∈ ∂f} logit_e`. Its *returned* score is built
separately from `logit − softplus(logit) = log p` on boundary bonds and `−softplus(logit) =
log(1 − p)` on internal bonds:

```text
prototype search  = Σ_all log(1−p) + Σ_{∂f} log(p/(1−p)) = B + I_f + I_f̄ = log P(m = ∂f)
prototype output  = Σ_{∂f} log p + Σ_{int} log(1−p)      = B + I_f
```

The search form is exactly `cut_masks`' mask log probability and exactly what
`_score_mask_log_probabilities` computes; the output form is the repo's beam score. They differ
by `I_f̄`, the same divergence derived algebraically above. The plan's fragment-probability
alignment resolved the inconsistency by making search follow the *output* convention, because
that is the score the downstream model consumes.

That single choice is the origin of everything below. Aligning the other way — output follows
search — would have been equally consistent, would have kept complements score-symmetric, and
would have placed the beam on `cut_masks`' objective exactly.

**The consequence is the cost.** Under mask scoring a fragment and its complement share the
mask and therefore have identical scores — which is what `mirror_symmetric` means in the
prototype. Mirrors are one XOR plus one connectivity check, run **once at the end on at most
`top_k` fragments**. Pruning by seed score before mirroring is then correct, because the
complement's score equals the seed's, so the audit's second finding cannot arise.

Changing the score broke that symmetry, which made pruning-before-mirroring a real defect,
whose fix moved complement generation into every growth round over unpruned candidates. The
ablation attributes 447 of the 493 ms regression to that pass and only 34 ms to stable sorting.
So the beam's dominant cost and its divergence from `cut_masks` are one decision, not two.

Reverting the score to the prototype's form would drop the internal-bond index arrays, make
complements score-free again, dissolve the second audit finding and most of the regression. The
price is precisely what `cut_masks` spends the charge head on: `f` and `f̄` become
indistinguishable, so the frozen charge head would be needed again to separate them.

Both audit defects are present in the prototype: `torch.topk` without a stable tie key, and
mirrors taken after the `rank < top_k` cap. Other divergences are per-molecule growth depth,
per-round dedup and top-K (absent in the prototype), the reserved precursor slot, vectorized
popcount, and the prototype computing `frag_mz`, `frag_num_h` and `frag_embed` inside the
search where the repo returns the sparse membership adapter and uses MASSter chemistry.

## The half-depth trick, and what pays for it

The search grows each molecule's atom sets only to `ceil(atom_count / 2)` rounds and recovers
everything larger as connected complements. This is correct and a genuine saving: every
bipartition has a side of at most half the atoms, so half-depth growth plus mirroring reaches
every connected fragment whose complement is also connected. It halves the growth rounds, and
each round costs `O(mols · beam · atoms²)` in the dense `bmm` update, so the saving is real.

The point worth recording is that **the saving is financed by the complement pass**, and the
complement pass is what the speed audit measured as dominant. The two are one design decision:
half-depth growth is only possible because complements are generated, and complements are
currently generated by dense reachability — for each recorded candidate, up to a graph-diameter
number of `bmm` iterations at `O(atoms²)` each, repeated every round.

Union-find changes that trade from expensive to free. Running components once on a candidate's
boundary cut set yields the fragment **and** every complement piece in the same pass, at
`O(bonds · α)`. There is then no separate complement step to pay for: the half-depth trick
keeps its 2× and loses its penalty. This is the strongest argument for the replacement, and it
is stronger than the "duplicated code" argument.

Part of that cost is synchronization rather than arithmetic, and the speed audit's operator
profile already measures it: `aten::equal` rises from 33 to 1,304 calls and CUDA stream
synchronizations from 1,175 to 3,048. Those `equal` calls are the reachability loop's
`th.equal(updated, reach)` break test, which returns a Python bool and so forces a
device-to-host synchronization on every iteration, inside every chunk, inside every round; the
audit names that host-visible equality test as a main contributor. `_search` synchronizes
further at `recorded.any()`, `int(counts.max().item())`, `int(keep_rank.max().item())` and at
each boolean-mask index. This is the failure mode the
[MASSter GPU audit](../../plans/masster/masster_gpu_audit.md) catalogues. It is independent of the
union-find question: an asynchronous fixed-iteration or count-based convergence test would
remove the syncs without changing the algorithm.

## Cost consequence

From the existing measurements, not re-run here:

- After the selection fixes, GPU candidate construction rises **238.7 → 732.1 ms (3.07×)** and
  forward/backward **744.4 → 1,252.3 ms** at batch 128, beam/target 256, boundary limit 5.
  **Complement connectivity dominates**; stable sorting is the smaller cost.
- Against the CUDA `cut_masks` backend on synthetic batches, the beam won only at 22–29 maximum
  atoms (41 vs 59 ms at batch 126; 57 vs 155 ms at batch 510) and lost at 49 maximum atoms
  (104 vs 76 ms; 235 vs 168 ms). Those numbers predate the selection fixes.
- Cost scales with the batch's **largest** molecule, because the dense tensors are
  `(mols, max_atoms, max_atoms)` and `(mols, beam, max_atoms)`. At batch 128 that samples the
  upper tail of the atom-count distribution every step, not the median.
- The speed audit's own sampled real batches have atom-count quartiles 14/19/26 with maxima of
  **76 and 95**. That is well above the 49-atom point where the beam already lost, which is
  evidence — on sampled batches, not a corpus census — that the regime where the beam wins does
  not hold for real unmerged batches.

So the reuse item and the speed item are the same item: replacing dense-reachability complement
checks with the union-find components MASSter already has is what decides whether the beam
is a cheaper route to nearly the same objective or a more expensive one.

## Limits

- Source and algebra review against `src/fragnnet/frag/masster_fast_gnn.py`, which is
  unchanged between `dca3d0f9` and the current tree, so every measurement describes the
  post-fix implementation.
- **No training run, optimizer step or validation-set quality evaluation was performed.** All
  measurements are candidate construction only: selectors are called directly on captured graphs
  and logits. Spectral accuracy is therefore **not** measured anywhere in this document, and no
  claim is made about predicted spectra.
- Timing figures for the 447 ms complement regression and the CUDA `cut_masks` comparison are
  quoted from the [speed audit](connected_fragment_beam_speed_2026_09_17.md) and the
  [plan](../../plans/masster/masster_fast_gnn_proposal.md) and inherit their contracts. Timings measured
  here are single-run CPU wall times, not synchronized benchmarks.
- One checkpoint, one epoch, one batch per scan; batches of 16 and 32 molecules. The reference
  universe is the union of the selectors' own pools, so it under-represents fragments none of
  them enumerate.
- The ring-gap magnitudes in the algebra section are worked examples at assumed `p_e`. Checks 1
  and 2 above remain wholly or partly unrun.
- New code added for this document: `scripts/debug/analyze_proposal_fragment_coverage.py`. No
  production code, model, config or test was modified.
