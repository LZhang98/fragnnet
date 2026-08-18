# MASShatter — One-Pass Bond-Set Fragmentation Plan

> **Status:** IN PROGRESS — FIRST TRAINABLE SLICE IMPLEMENTED &nbsp;·&nbsp; **Updated:** 2026-08-15
>
> Replace fixed-depth D3/D4 fragment enumeration at inference with one molecular
> encoding pass, independent bond-break probabilities, top-K cut-mask search,
> component-level charge retention, and a conditional hydrogen-transfer head.

## Objective

Predict a break probability for every molecular bond in one model pass and
translate the highest-probability bond-cut masks directly into a flat set of
charged fragment-ion candidates. Generation is controlled by probability and a
candidate budget, not by a fixed fragmentation depth.

The model's canonical public name is **MASShatter**. Code uses
`MassShatterModel`, `MassShatterPL`, and `model_type: mass_shatter`; generic
bond-mask and neural-head utilities retain descriptive names because they are
reusable components rather than the complete model.

```
precursor molecular graph
  -> atom/bond GNN embeddings (one pass)
  -> one break probability per bond
  -> top-K global cut masks
  -> connected components
  -> component charge-retention probability
  -> conditional (ion mode, delta-H) probability
  -> fragment-ion masses and peak probabilities
```

This permits a selected mask to contain one, four, or more broken bonds without
introducing D1/D3/D4/D5 as model concepts. D3 and D4 remain evaluation baselines
and may optionally supply warm-start labels, but they are not part of inference.

## Scope and non-goals

- Generate candidates by deleting existing precursor bonds. The first version
  does not model products that require forming a new heavy-atom bond. Section
  6.1 specifies a planned template channel for their spectrum-level support.
- Assume independent Bernoulli bond-breaking probabilities in the first
  version. Pairwise/concerted corrections are an ablation only after the
  independent model is measured.
- Model singly charged fragment ions first. Multiply charged products require a
  different charge-allocation model.
- Return a flat deduplicated fragment-ion set. Parent masks can be retained as
  provenance, but the intensity model must not require a fragmentation tree.
- Separate formation, charge retention, ion formula, and final detected
  intensity. They are related probabilities, but they are not the same event.

## 1. One-pass bond-break model

For a molecule with `B` bonds, the bond head returns one logit and independent
sigmoid probability per bond:

```
l_b = BondHead(z_b, molecular_context, experiment_context)
p_b = sigmoid(l_b),  b = 1, ..., B
```

This is **one output per bond**, not a one-hot softmax over bonds. A softmax would
assert that exactly one bond breaks and could not represent multi-bond masks.

`z_b` is built from the two endpoint atom embeddings and the original bond
features. A dedicated bond/line-graph GNN is optional: first compare it against
the simpler endpoint encoder
`MLP([h_u + h_v, |h_u - h_v|, edge_features])`. The bond GNN must earn its
additional cost on held-out ranking and fragment coverage.

If the output is intended to predict spectrum-specific peaks, experiment context
includes precursor/adduct type, polarity, collision energy, instrument, and
fragmentation mode. If one candidate set must be shared across spectra, keep the
bond head molecule-only and apply experimental conditioning in the charge,
H-transfer, and intensity heads instead.

## 2. Probability of a complete cut mask

Let `C` be a multi-hot subset of broken bonds. Under the independent Bernoulli
model:

```
log P(C)
  = sum_{b in C}     log p_b
  + sum_{b not in C} log(1 - p_b)
```

`C` represents the **net set of absent precursor bonds** in the final product
state. It does not assert that all bonds broke simultaneously, and it does not
identify an ordering among sequential cleavages. Different mechanisms with the
same final atom/edge state are deliberately collapsed in this first model.

All probability calculations stay in log space. Raw mask probabilities can be
extremely small and are not suitable for direct thresholding.

For ranking masks within one molecule:

```
log P(C)
  = sum_b log(1 - p_b) + sum_{b in C} logit(p_b)
```

The first term is constant across masks, so relative ranking can use the sum of
selected bond logits. Absolute calibrated probabilities still use the complete
Bernoulli expression.

### 2.1 Exact top-K mask enumeration under the independent model

The top masks do not require enumerating all `2**B` subsets.

1. Construct the MAP mask `M`, where `M_b = 1` if `l_b > 0`.
2. Flipping bond `b` away from `M` costs `d_b = |l_b|` in log probability.
3. Sort the nonnegative flip costs.
4. Use a min-heap K-smallest-subset-sums algorithm to enumerate masks in exact
   descending probability order.

The expected bookkeeping cost is approximately:

```
O(B log B + K log K)
```

The queue mixes every cut count automatically. A cut-4 mask appears before a
cut-1 mask whenever its total learned score is higher. No cut cardinality is
specified in advance. When `M` is nonempty, track the actual cut count from
`M XOR flips`; the number of flips is not necessarily the number of broken
bonds.

#### Why the heap enumeration is exact

The Bernoulli logit is the log-odds of breaking a bond:

```
l_b = log(p_b / (1 - p_b))
```

If `l_b > 0`, the MAP decision is broken and changing it to intact changes the
mask log probability by `-l_b`. If `l_b <= 0`, the MAP decision is intact and
changing it to broken changes the log probability by `l_b`. In both cases a
flip subtracts `abs(l_b)`. Independence makes the penalties additive, so every
mask `C` can be written uniquely as the MAP mask XOR a flip set `S`, with:

```
log P(C) = log P(M) - sum_{b in S} abs(l_b)
```

Ranking masks by decreasing probability is therefore the same problem as
ranking subsets of the sorted nonnegative flip costs
`d_0 <= d_1 <= ... <= d_(B-1)` by increasing subset sum.

The subset-sum search starts at `{0}`. For a subset whose largest sorted-cost
position is `i`, it creates two children:

```
append:   S union {i + 1}
replace:  (S without {i}) union {i + 1}
```

This tree contains every nonempty subset exactly once. For any subset with
largest position `j`, its unique parent is obtained by removing `j` when it
also contains `j - 1`; otherwise its parent is obtained by replacing `j` with
`j - 1`. Append increases cost by `d_(i+1)`, and replace increases cost by
`d_(i+1) - d_i`; both are nonnegative because the costs are sorted.

The min-heap holds the current tree frontier. Since descendants cannot be
cheaper than their ancestors, removing the cheapest frontier entry yields the
next globally smallest subset sum and therefore the next globally most
probable cut mask. An unseen cheaper subset cannot be skipped: an unexpanded
ancestor on its path would be no more expensive and would already be present
in the heap.

This guarantee depends on the independent-bond factorization. Pairwise or
higher-order bond interactions generally make flip penalties non-additive, so
this subset-sum heap would no longer provide exact probability order.

#### Worked example

Consider three bonds with logits:

```
bond:          0       1       2
logit:       2.0    -1.0     0.2
MAP state:   cut   intact     cut
flip cost:   2.0     1.0     0.2
```

The MAP mask is therefore `{0, 2}`. Its log probability is:

```
log P({0, 2})
  = log sigmoid(2.0) + log sigmoid(1.0) + log sigmoid(0.2)
  = -1.0383
```

After sorting, the flip positions are bond 2 at cost `0.2`, bond 1 at cost
`1.0`, and bond 0 at cost `2.0`. A flip toggles the corresponding MAP state;
it does not always mean cutting a bond. For example, flipping bond 2 changes it
from cut to intact, while flipping bond 1 changes it from intact to cut.

The resulting masks begin in this exact order:

| Rank | Flipped bonds | Total flip cost | Resulting cut mask | Log probability |
|------|---------------|----------------:|--------------------|----------------:|
| 1 | none | 0.0 | `{0, 2}` | -1.0383 |
| 2 | `{2}` | 0.2 | `{0}` | -1.2383 |
| 3 | `{1}` | 1.0 | `{0, 1, 2}` | -2.0383 |
| 4 | `{1, 2}` | 1.2 | `{0, 1}` | -2.2383 |
| 5 | `{0}` | 2.0 | `{2}` | -3.0383 |

For the heap mechanics, after `{2}` is removed from the heap, its two children
are `{2, 1}` with cost `1.2` (append) and `{1}` with cost `1.0` (replace).
The min-heap removes `{1}` next because `1.0 < 1.2`, which gives the third mask
above. Continuing the same two-child expansion enumerates all remaining masks
by increasing total flip cost. `top_k_cut_masks` simply stops after `K` masks.

## 3. Translate a cut mask into fragments

For every selected mask `C`:

1. Remove its selected molecular bonds.
2. Compute the connected components of the remaining graph.
3. Represent each component by its precursor atom-membership mask.
4. Discard a mask if it creates no new component, such as a lone ring opening,
   unless ring-opened bond state is deliberately included in the product
   representation.
5. Deduplicate identical component atom masks across cut masks.

The same component can be produced by several complete cut masks because bonds
outside the component may break or stay without changing it. Therefore the
component's formation probability is a marginal:

```
P_form(F) = sum_{C: F in components(G \ C)} P(C)
```

Accumulate these contributions with `logsumexp`.

This marginal is exact only if every contributing mask is included. Top-K mask
enumeration gives a controlled approximation. Exact top-K masks are easy under
the independent model; exact top-K **marginal fragments** are harder because an
unseen tail of masks can add probability to an already-seen fragment.

### 3.1 Stopping rules

Enumerate masks in descending probability until the first applicable condition:

- `max_masks` is reached, initially compare 100 and 500;
- `max_unique_fragments` is reached;
- the next mask is more than `delta_logp` below the best mask;
- an optional retained-probability-mass target is reached;
- the per-molecule time or memory budget is reached.

Prefer a relative log-probability threshold over a fixed raw probability because
the latter changes systematically with molecule size.

If duplicate masks consume much of the budget, continue past `max_masks` until
`max_unique_fragments` or another safety limit is reached. Record both counts so
the cost is visible.

## 4. Charge retention / ionization chance

Bond breaking produces neutral connectivity components; it does not determine
which component becomes an observed ion. For an already-ionized precursor this
term is more precisely **charge retention**, although the model can be viewed as
predicting fragment ionizability.

Add an atom-level charge-affinity head conditioned on precursor/adduct type and
polarity:

```
a_i = ChargeHead(h_i, precursor_type, polarity)
```

For component `F_j` from cut mask `C`, aggregate its possible charge sites:

```
z_charge(F_j) = logsumexp_{i in F_j} a_i
```

For singly charged products, normalize across all components from the same cut
mask:

```
P_charge(F_j | C, precursor_type)
  = softmax_j(z_charge(F_j))
```

This makes complementary products compete for one charge and stays consistent
when a mask produces more than two components. Across an ensemble of precursor
ions, different events can still produce peaks for both complementary sides.

An atom-affinity head is preferred over independent `charge-left` /
`charge-right` outputs on every boundary bond. Per-bond directions can conflict
when a fragment has several boundary bonds, whereas component-level charge
competition yields one coherent probability.

## 5. Conditional hydrogen-transfer and ion-mode head

After choosing a charged component, predict its observable ion formula. The
H-transfer head is conditional on the component, its complement, the complete
cut mask, and experimental context:

```
P(mode, delta_H | F, C, charged, precursor_type, CE)
```

Recommended inputs are:

- pooled atom embeddings inside `F`;
- pooled atom embeddings outside `F`;
- pooled embeddings of the bonds crossing `F`'s boundary;
- cut count and simple component formula features;
- precursor/adduct type, polarity, collision energy, instrument, and
  fragmentation mode when available.

Represent atom/component and boundary-bond/component membership as sparse index
pairs and pool with `index_add`; do not materialize a
`num_components x num_batch_atoms` boolean matrix. When components from several
cut masks are batched, carry a required component-to-cut-mask group index so
charge retention is normalized independently inside every cut mask.

The initial categorical output is the cross product of:

- ion mode: CRF and, when enabled, CMF;
- `delta_H` in the configured finite range, for example `[-4, ..., +4]`.

Mask chemically invalid categories before softmax. At minimum:

- the resulting formula must have nonnegative element counts;
- H removal cannot exceed the component's available hydrogens;
- positive H transfer cannot exceed the complement's transferable hydrogen
  supply under the selected ion mode;
- adduct/mode combinations absent from the formula tables are invalid;
- current CRF/CMF conventions in `src/fragnnet/utils/formula_utils.py` apply;
- for `[M+H]+` and `[M-H]-`, CMF `delta_H != 0` is invalid because those rows
  duplicate or contradict the existing direct-charge convention.

The ion formula is:

```
formula_ion
  = formula(F)
  + mode_specific_precursor_correction
  + delta_H * H
```

Making `(mode, delta_H)` a joint conditional head is intentional. Charge
retention, carrier loss, and hydrogen transfer are coupled; independent heads
would assign probability to incompatible combinations.

New-heavy-bond-forming rearrangements remain out of scope. An H-transfer head can
change formula/mass, but it cannot make a disconnected atom mask into a product
that requires a new C-C, C-N, or C-O bond.

### 5.1 H-transfer probability normalization

For every charged component, the head returns logits with shape:

```
(num_components, num_ion_modes, 2 * max_h_transfer + 1)
```

With CRF and CMF enabled and `max_h_transfer = 4`, this gives 18 raw channels:
the cross product of two ion modes and `delta_H = [-4, ..., 0, ..., +4]`.
`delta_H = 0` is the no-transfer outcome, so a separate binary H-transfer gate
is unnecessary.

Apply the chemical-validity mask before flattening the mode/H axes and taking
one softmax per component:

```
log P(mode, delta_H | F, C, charged, context)
  = log_softmax(mask_invalid(joint_logits))[mode, delta_H]
```

Thus the probabilities of all valid `(mode, delta_H)` outcomes sum to one for
each charged component. Do not normalize H-transfer independently of ion mode
or multiply it by a second transfer/no-transfer probability.

The probability of one complete latent pathway is:

```
log P(path)
  = log P(C)
  + log P_charge(F | C, precursor_type)
  + log P(mode, delta_H | F, C, charged, context)
```

For example, a cut-mask probability of `0.08`, charge-retention probability of
`0.70`, and conditional `(CRF, delta_H=+1)` probability of `0.25` give a pathway
probability of `0.014`. If another retained cut mask produces the same final ion
with probability `0.006`, its marginal ion probability is `0.020`. Perform this
sum with `logsumexp`, as specified below, rather than adding raw probabilities
in the implementation.

## 6. Complete fragment-ion probability

For ion candidate `(F, mode, delta_H)`, marginalize over retained cut masks:

```
P_ion(F, mode, delta_H)
  = sum_{C: F in components(G \ C)}
      P(C)
      * P_charge(F | C, precursor_type)
      * P(mode, delta_H | F, C, charged, context)
```

Implement the sum as `logsumexp` over pathway log probabilities. Candidates that
resolve to the same ion formula/mass can be aggregated again before the spectrum
loss.

`P_ion` is not automatically the final peak intensity. Detection efficiency,
secondary decay, isotope abundance, and instrument response can change intensity.
The first comparison should test both:

1. use normalized `P_ion` directly as predicted intensity;
2. feed `log P_ion` plus component/context features into a small intensity head.

Retain the intensity head only if it gives a reproducible held-out improvement.

### 6.1 Planned template channel for new-bond products

Ring contractions and other rearrangements may form a new heavy-atom bond, so
their product connectivity cannot be obtained from `components(G \ C)`. A
general graph-edit decoder is not required to predict their formula, m/z, and
intensity: bond formation changes connectivity but does not by itself change
elemental composition. Represent a template product by its precursor atom mask
even when those atoms are disconnected in the precursor graph.

Reuse the existing SMARTS prepass conventions in
`src/fragnnet/frag/smarts_prepass.py`:

- `SmartsFragRule` already specifies cut bonds and merges disconnected
  components into one fragment;
- `NeutralLossRule` already represents a new-bond product as all precursor
  atoms except the neutral-loss atoms;
- validated all-carbon and heteroatom Retro-Diels--Alder SMIRKS are covered in
  `tests/test_rearrangement_smarts.py`.

Precompute template matches during molecular preprocessing rather than running
RDKit SMARTS matching in the training hot path. Each match should retain:

- the precursor bond indices deleted by the mechanism;
- the product and complementary atom masks;
- the reaction-rule identifier;
- allowed or required `delta_H` values;
- optionally, formed-bond endpoints for provenance and future structural use.

Feed these atom partitions through the existing component/complement pooling,
charge-retention, ion-mode/H-transfer, mass, and intensity heads. The first
version does not need to materialize the new bond because no downstream model
consumes product connectivity.

Normal cleavage and a rearrangement can share the same deleted-bond mask. Do not
append the rearrangement as an unrelated Bernoulli mask: that double-counts the
mask probability and invalidates retained/out-of-support mass accounting.
Instead introduce a mechanism variable `R`, including the ordinary connected-
component outcome, and normalize mechanism logits within each retained cut
mask:

```
P(path)
  = P(C)
    * P(R | C, precursor, context)
    * P_charge(F | C, R, precursor_type)
    * P(mode, delta_H | F, C, R, charged, context)
```

This preserves the independent-Bernoulli top-K proposal and its out-of-support
probability while allowing the candidate/intensity model to choose between
ordinary cleavage and template-defined atom partitions. Add a learned rule
embedding to the mechanism, charge, and ion heads; do not give every matched
template an unconditional probability floor.

This extension predicts spectrum-level outcomes, not arbitrary product
structures. Isomeric ring contractions with the same atom set and formula remain
indistinguishable by m/z. If subsequent fragmentation depends on the newly
formed connectivity, templates must provide those downstream products directly;
a general sequential graph-edit model is a separate, substantially larger
extension.

Verify before adoption:

- atom conservation, product/complement partitioning, and template
  deduplication;
- mechanism probabilities sum to one within each retained cut mask;
- template candidates do not alter retained or out-of-support mask mass;
- candidate and intensity-weighted recall improve on targeted rearrangement
  classes at a matched realized candidate budget;
- held-out cosine/JSS improve without increasing false candidates on molecules
  where a SMARTS pattern matches but the rearrangement peak is absent.

## 7. Training supervision

The difficult part is not top-K mask enumeration; it is identifying which bonds
explain a spectrum. Experimental peaks usually provide m/z and intensity, not a
unique atom mask or cut mask.

### 7.1 Preferred staged training

1. **Mechanics tests with synthetic logits.** Verify top-K ordering against brute
   force on small graphs, connected-component conversion, deduplication, and
   probability aggregation before training any model.
2. **Optional supervised warm start.** If trusted fragment atom masks are
   available, supervise boundary-bond breaks, charge retention, and
   `(mode, delta_H)` directly. Existing D3/D4 artifacts may be used as temporary
   teachers, but the trained generator must not depend on them at inference.
3. **Spectrum-level fine-tuning.** Generate top-K latent ion candidates, map them
   to m/z, aggregate candidates explaining the same peak, and optimize the normal
   spectrum loss plus coverage/calibration diagnostics.

Without warm-start labels, top-K selection is discrete and missed candidates
receive no gradient. A fully latent cold start therefore needs an exploration
mechanism such as tempered sampling or stochastic masks; deterministic top-K
alone can lock into initially incorrect bonds.

### 7.2 Additive bounded-cardinality support augmentation

This mechanism is **stochastic support augmentation**: it broadens the candidate
set the spectrum loss can assign probability to. It is not importance sampling,
it does not estimate the full mask marginal, and retained masks carry no
correction weight. Its proposal distribution has no tractable per-mask
likelihood, which constrains any policy built on top of it — see
[masshatter_rl_exploration_plan.md](masshatter_rl_exploration_plan.md).

**Implemented 2026-08-16.** The default `bounded_additive` mode preserves all
`mass_shatter_max_masks` deterministic masks and adds up to
`mass_shatter_explore_masks` stochastic masks during training. Validation, test,
and prediction retain exact deterministic top-K. Historical mixed-top-K configs
explicitly select `bernoulli_replace` so their original replacement semantics
remain reproducible.

The bounded proposal separates how many bonds break from which bonds break. It
first draws a cut count from
`mass_shatter_exploration_cut_count_weights`, whose initial 2--6 distribution is
35/30/20/10/5 percent. It then samples that many distinct bonds with Gumbel
top-k over the temperature-scaled live bond logits. Empty and single-cut masks
remain forced separately. Duplicates are rejected, unsuccessful exploration
slots are not refilled, and the actual exploration count and mean cut count are
logged.

Temperature changes only which bonds are proposed at a fixed sampled cut count.
Every retained mask's log probability is recomputed from the original,
untempered live logits so downstream probabilities preserve their existing
meaning and gradients. Mask selection remains detached and non-differentiable:
this phase broadens candidate support but is not policy-gradient RL.

Use bounded selection only for training. Validation, test, and prediction use
the existing deterministic exact top-K path. The deterministic control and
matched K256+64 experiment are
`config/masshatter/nist20_merged_mask256_gine_efa384_bondcentric_a2_b3.yml` and
`config/masshatter/nist20_merged_mask256_gine_efa384_bondcentric_a2_b3_explore64.yml`.

Implement and verify in this order:

1. Add seeded, per-precursor stochastic mask proposals and deterministic
   deduplication/refill to candidate construction.
2. Test that the exact exploitation masks are always retained, the learned-mask
   count is bounded by `K`, forced masks remain present, fixed seeds reproduce
   training candidates, and evaluation is seed-independent.
3. Verify that retained mask log probabilities still backpropagate through the
   original logits and that proposal temperature does not alter those values.
4. Compare deterministic top-K against mixed top-K at matched actual mask and
   component budgets. Report candidate peak recall, intensity-weighted recall,
   cosine/JSS, unique fragments, runtime, memory, and variability across at
   least three seeds.

Adopt mixed selection only if it improves held-out candidate coverage and final
spectrum metrics without relying on a larger realized candidate budget. If it
only increases duplicate or low-value masks, test a diversity-aware proposal
before considering a sequential or RL policy.

### 7.2.1 Candidate-selection performance plan

**Observed 2026-08-15.** The initial L40S runs report 11--16% GPU utilization
and 4--22 GiB allocated VRAM. The immediate configuration change doubles the
physical training batch while preserving the effective batch of 128: use
`16 * 8` for 256-mask runs and `8 * 16` for the 512-mask control. Selected-mask
probability scoring is also batched, rather than allocating and scoring one
GPU boolean mask at a time.

The remaining bottleneck is exact candidate selection: each precursor's live
bond logits and endpoints are copied to CPU, then Python heap enumeration and
connected-component translation run independently for every selected mask.
Do not cache logits or mask probabilities: they change every optimization step
and must retain their gradient. Optimize the static topology work instead.

1. **Topology cache first.** Cache
   `(molecule topology, broken_bond_tuple) -> component atom memberships` in a
   bounded LRU cache. The empty and single-bond masks, which
   are forced candidates on every visit, populate the cache on a molecule's
   first visit. Cache values remain CPU
   Python/index data; construct the final sparse device tensors once per batch.
   Record cache hit rate, entries, and bytes. Validate cached and uncached
   candidates have identical sparse indices and mask log probabilities.
2. **Profile before a native implementation.** Measure separate wall time for
   device-to-host transfer, top-K enumeration, component translation, cache
   lookup, sparse-tensor construction, forward/backward, and data loading. Run
   the profile at the production 256 and 512 budgets with the larger batches.
   Adopt each optimization only if examples/second improves at unchanged mask
   budget and coverage.
3. **Native CPU path if topology work remains dominant.** Port the exact
   heap/subset-sum enumeration and union-find component translation together to
   a PyTorch C++ extension. This avoids Python loop overhead while preserving
   the exact independent-Bernoulli ordering. Cython is a viable fallback for
   the same CPU routines, but it does not integrate device tensors as cleanly
   and cannot remove the synchronization point.
4. **GPU path only after the above gate.** A GPU candidate generator requires a
   custom CUDA/Triton implementation for variable-size K-best subset selection
   and masked connected components; there is no built-in PyTorch equivalent.
   Keep the exact CPU path as the reference oracle and require identical masks
   on small graphs. A tensorized approximate selector is a separate modeling
   change and must be evaluated for coverage/calibration, not presented as a
   transparent acceleration.

### 7.3 Loss and evaluation units

- Split by molecule or scaffold, never by spectrum alone.
- Aggregate all latent explanations of the same observed peak with `logsumexp`;
  do not assign a hard fragment label when multiple atom masks share a formula.
- Report metrics by cut count, ring/non-ring status, precursor type, collision
  energy, and scaffold novelty.
- Keep calibration metrics separate for bond breaks, charge retention,
  H-transfer, and final peak probability.

## 8. Implementation stages and verification gates

### Stage 1 — Probability translator, no learned model — IMPLEMENTED 2026-08-14

Implement a standalone top-K mask enumerator and mask-to-ion translator using
provided synthetic bond, atom-charge, and H-transfer logits.

Implemented the model-independent portion in
`src/fragnnet/frag/bond_mask_generator.py`: lazy exact mask ranking, bounded
top-K collection, connected-component translation, and truncated fragment
marginal aggregation. `tests/test_bond_mask_generator.py` verifies the ranking
against brute force through eight bonds and covers unrestricted cut counts,
ring cuts, duplicate marginal paths, extreme logits, and input validation.
Charge retention and H-transfer remain Stage 2/3 work.

Bond endpoints use PyG's `(2, num_bonds)` layout across the learned head and
translator. The initial mask budget may be extended when high-ranked masks only
open rings or duplicate previously generated atom subsets, with a separate hard
enumeration cap.

Verify:

- exact top-K mask equality with brute force for small `B`;
- no fixed cut-count/depth assumption;
- components match a reference graph implementation;
- duplicate fragment and ion probabilities equal direct probability sums;
- stable behavior for probabilities near 0 and 1;
- CPU determinism and bounded memory at `K=500`.

### Stage 2 — MASShatter bond and charge heads

Add the one-pass molecular encoder, endpoint bond baseline, optional line-graph
bond GNN, and atom charge-affinity head.

**Initial integration implemented, 2026-08-14:**
`src/fragnnet/model/bond_fragment_heads.py` now contains the symmetric endpoint
bond baseline, atom charge-affinity head, differentiable selected-mask
log-probability evaluation, component charge competition, and component/boundary
pooling. Charge competition is segmented by cut mask, and all component pooling
uses sparse membership pairs. `MassShatterModel` reuses FraGNNet's molecular GNN
as an unpooled `precursor_encoder`; molecular PyG graphs now carry canonical
chemical bonds and atom chemistry fields. `MassShatterPL` uses `SpecMolDataset`
and `SpectrumPL` rather than the D3/D4 dataset and training wrapper.

**Bond-focused ablation implemented, 2026-08-16:** setting
`mass_shatter_bond_num_layers` above zero replaces only the endpoint cleavage
head with explicit bond states and alternating bond-to-atom/atom-to-bond
updates. Bonds communicate when they share an atom, without materializing a
line graph, and endpoint-order invariance is preserved. A value of zero keeps
the original endpoint baseline. The charge/ion heads and independent-Bernoulli
top-K mask ranking remain unchanged. The controlled K256 GINE-EFA experiment is
`config/masshatter/nist20_merged_mask256_gine_efa384_bond2.yml`.

**Architecture direction from the initial W&B runs, 2026-08-16:** increasing
the atom-centered GINE encoder from 256/4 layers to 384/5 layers did not improve
validation cosine, while GINE-EFA reached the best observed score and converged
in fewer epochs. Mixed-mask exploration improved candidate weighted recall but
did not improve cosine. These results favor a more bond-focused encoder over a
larger generic atom transformer. The next controlled architecture experiment
should maintain explicit learned bond states, update adjacent bonds through a
line graph or edge-aware attention, alternate atom-to-bond and bond-to-atom
updates, and condition bond attention on acquisition context. Keep the current
independent-Bernoulli mask ranking and the charge/ion heads unchanged for this
comparison so any gain can be attributed to the bond representation. Only if
that model improves held-out results should correlated multi-bond scoring be
tested; pairwise or autoregressive cut dependencies change the exact top-K
semantics and are a separate, higher-risk experiment.

Verify:

- bond predictions are invariant to endpoint ordering;
- charge probabilities across components sum to one;
- the bond GNN beats the endpoint baseline before retaining it;
- held-out bond ranking and fragment coverage beat random mask selection.

### Stage 3 — H-transfer / ion-mode head

Add component/complement/boundary pooling and the joint categorical head.

**Initial integration implemented, 2026-08-14:** the same module contains the
joint ion-mode/H-transfer head and masked channel normalization. MASShatter now
constructs per-adduct chemical-validity masks and monoisotopic CRF/CMF masses,
including explicit H-transfer channels. Isotope envelopes remain out of scope
for the first training slice.

Verify:

- invalid formula channels receive zero probability;
- CRF/CMF masses match `formula_utils.py` exactly;
- known complementary components can receive different H-transfer
  distributions;
- per-adduct and per-polarity calibration is reported.

### Stage 4 — Flat fragment-ion integration

Return a flat fragment-ion batch containing atom masks, cut-mask provenance,
formula/mass, generation log probability, charge probability, ion-mode/H-transfer
probability, and combined log probability. Do not require DAG depth or DAG edges.

**Initial integration implemented, 2026-08-14:** sparse mask/component batching
preserves `(molecule, cut mask, component)` provenance and emits
`pred_mzs`, `pred_logprobs`, `pred_batch_idxs`, and truncated OOS probability for
`SpectrumPL`. Training uses the same sparse cross-entropy implementation as
FraGNNet. Empty and all single-bond masks are forced exploration candidates in
addition to learned top-K masks, preventing the zero-signal random-initialization
failure observed with deterministic top-K alone. Consequently, `max_masks` is
the learned ranking budget; the actual count can be larger by the single-bond
exploration set and is logged explicitly.

The first production-shaped configuration is
`config/nist20_merged_inchikey/mass_shatter/s0.yml`: K=100, CRF/CMF channels,
H-transfer range -4 through +4, and FraGNNet's sparse Gaussian cross-entropy.
Training logs intensity-weighted candidate recall, retained exact-mask
probability, selected mask count, and component pathway count so support errors
can be separated from probability-allocation errors.

Verify at matched candidate budgets against D3 and D4:

- formula recall and intensity-weighted peak recall;
- downstream cosine/JSS metrics;
- unique fragments per evaluated cut mask;
- runtime and peak memory, including combinatorial outlier molecules;
- recall beyond four broken bonds;
- stability across at least three seeds.

The design advances only if it beats both random-mask selection and the existing
D3 candidate set at a matched fragment budget on held-out scaffolds. D4 is the
higher-recall reference, not an inference dependency.

## 9. Review findings and unresolved risks

1. **Top-K masks are tractable; literal all-fragment marginalization is not.**
   The independent model permits efficient exact mask ranking, but fragment
   probabilities remain truncated approximations unless every mask is summed.
2. **Deep masks may be systematically under-ranked.** Products of sub-unit bond
   probabilities naturally penalize cut count. Measure calibration by cut count
   before adding a temperature or learned cut-count prior; either adjustment
   changes the probabilistic interpretation.
3. **Independent bonds cannot express concerted chemistry.** Correlated ring
   release and rearrangements may need pairwise factors. Do not add them until
   the independent baseline identifies a measurable failure class.
   The larger limitation remains that independent bond probabilities cannot learn
   correlations between cuts; an autoregressive or sequential policy would address
   that later.
4. **Static charge sites are an approximation.** Mobile-proton behavior can
   relocate charge after cleavage. Component/complement context in the
   H-transfer head helps, but the atom charge-affinity head may still need a
   component-conditioned correction.
5. **Top-K training has a cold-start problem.** Deterministic masks omitted from
   the candidate set receive no spectrum-level gradient. Warm-start supervision
   or explicit exploration is required.
6. **Mask probability is not peak intensity.** A small post-generation intensity
   head may be necessary, but it must not become another expensive per-fragment
   model without a measured benefit.
7. **New-bond rearrangements are unreachable in the current action space.** The
   planned atom-mask/template mechanism in Section 6.1 can support their direct
   spectrum-level products. Arbitrary product structures and subsequent
   connectivity-dependent fragmentation still require a learned edit channel.
8. **MAP masks with many `p_b > 0.5` are a calibration warning.** The K-best
   algorithm remains correct, but a heavily cut MAP state is chemically suspect
   and should be monitored explicitly.
9. **Charge pooling can carry a component-size bias.** If atom affinities are
   similar, `logsumexp` assigns more charge probability to a component merely
   because it contains more atoms. Compare it against normalized log-mean-exp or
   learned attention pooling; retain `logsumexp` only if charge-retention
   calibration supports the implied multiple-site interpretation.
10. **Net cut masks discard cleavage order.** Two sequential pathways and one
    concerted pathway can produce the same mask but need not have the same
    probability or H-transfer behavior. This is the cost of removing the tree;
    measure errors by cut count and known concerted chemistry before considering
    any path-sensitive extension.

## Related infrastructure

- `src/fragnnet/utils/formula_utils.py` — precursor, CRF, and CMF formula/mass
  conventions that the ion-mode/H-transfer head must reproduce.
- `src/fragnnet/frag/multi_cut_bfs.pyx` — reference connected-component and ring
  behavior for tests only; it is not the proposed inference engine.
- `src/fragnnet/model/fragnnet_model.py` — existing molecular embeddings,
  fragment formula channels, and spectrum aggregation conventions.
- `src/fragnnet/iceberg/pl_model.py` — precedent for learned probabilistic
  generation, although its autoregressive atom-removal mechanism is not reused.
- `docs/plans/dag_full_recall_tractability_plan.md` — records why static bond
  ranking inside a fixed DAG did not solve the earlier node-budget problem. The
  present proposal instead ranks complete cut masks and directly generates a
  flat fragment set.
