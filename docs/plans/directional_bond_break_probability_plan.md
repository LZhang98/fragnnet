# MASShatter — One-Pass Bond-Set Fragmentation Plan

> **Status:** IN PROGRESS — FIRST TRAINABLE SLICE IMPLEMENTED &nbsp;·&nbsp; **Updated:** 2026-08-14
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
  does not model products that require forming a new heavy-atom bond.
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

### 7.2 Loss and evaluation units

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
and `SpectrumPL` rather than the D3/D4 dataset and training wrapper. The optional
line-graph bond GNN remains an ablation.

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
7. **New-bond rearrangements are unreachable.** SMARTS/reaction templates or a
   separate learned edit channel are required for those products.
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
