# MASShatter — One-Pass Bond-Set Fragmentation Plan

> **Status:** IMPLEMENTED CORE WITH ACTIVE EXPERIMENTAL VARIANTS &nbsp;·&nbsp; **Updated:** 2026-09-21
>
> The fixed-depth D3/D4 inference path has been replaced in MASShatter by one-pass
> precursor encoding, probability-ranked candidate generation, chemically valid ion
> channels, and candidate-conditional intensity prediction. This document retains
> its original design and staged-history sections below; the current implementation
> snapshot is authoritative for present behavior.

## Objective

Predict a break probability for every molecular bond in one model pass and
translate the highest-probability bond-cut masks directly into a flat set of
charged fragment-ion candidates. Generation is controlled by probability and a
candidate budget, not by a fixed fragmentation depth.

The model's canonical public name is **MASShatter**. Code uses
`MassShatterModel`, `MassShatterPL`, and `model_type: mass_shatter`; generic
bond-mask and neural-head utilities retain descriptive names because they are
reusable components rather than the complete model.

## Current implementation snapshot (source-verified 2026-09-21)

The production code is `MassShatterModel` in
[`src/fragnnet/model/mass_shatter_model.py`](../../../src/fragnnet/model/mass_shatter_model.py)
and its Lightning/loss integration is `MassShatterPL` in
[`src/fragnnet/pl_model/mass_shatter_pl.py`](../../../src/fragnnet/pl_model/mass_shatter_pl.py).
The resolved configuration contract is declared in
[`config/template.yml`](../../../config/template.yml). A run's actual behavior is
its resolved configuration; the defaults below are not a claim about every
historical checkpoint.

```
precursor graph + spectrum metadata
  -> molecular GNN (optionally a separate proposal GNN; optional context at GNN input)
  -> one or more Bernoulli bond-logit fields + normalized mixture weights
  -> probability-ranked cut masks (default) or experimental connected-fragment beam
  -> connected components / candidate atom sets
  -> charge retention and chemically valid CRF/CMF, delta-H channels
  -> duplicate-pathway marginalization in unique-fragment mode
  -> candidate intensity transformer
  -> sparse m/z prediction with normalized probabilities or independent intensities
```

### Implemented mainline

- **Proposal.** `cut_masks` is the default proposal method. It ranks complete
  Bernoulli cut masks exactly within the requested prefix; `proposal_num_mixtures`
  can represent a normalized mixture of Bernoulli fields. The candidate builder
  supports CPU by default and an opt-in CUDA backend for its documented restricted
  mode. `connected_fragment_beam` is implemented but remains experimental; it has
  different reachability and probability semantics and is not a drop-in scientific
  replacement for cut masks.
- **Candidate identity.** Ordinary mode retains ranked mask/component pathways.
  In unique-fragment mode, enumeration continues toward a distinct atom-set target,
  and duplicate pathways are combined with `logsumexp` before intensity scoring.
  Candidate count, target fill, pathway duplication, cut-count, coverage, and timing
  diagnostics are returned by the model.
- **Chemistry.** Components receive charge-retention scores and all chemically valid
  CRF channels plus enabled CMF and hydrogen-transfer channels. The current CMF
  contract emits the full configured valid H-transfer ladder, suppresses degenerate
  duplicate CRF/CMF channels, and excludes in-source-loss channels. SMARTS templates
  are an optional additional candidate source.
- **Intensity.** A candidate-set transformer scores the generated ion channels.
  It supports induced attention by default and full attention only with the bounded
  unique-fragment representation. Candidate mass and charge features are enabled by
  the template defaults. The obsolete atom decoder is disabled and rejected by the
  Lightning-module configuration validation.
- **Output and loss.** The legacy/default `probability` output normalizes over the
  generated candidate set and returns `pred_logprobs`; unmatched observed signal is
  a coverage diagnostic, not an OOS class. The optional `independent` output returns
  nonnegative `pred_intensities` and requires the compatible unique-fragment,
  proposal-selection-only setup. MASShatter accepts the historical CE/JSS loss paths
  and the separately named `conditional_cross_entropy` path; the latter is currently
  implemented in the working tree and has not yet been established as a trained-model
  improvement.

### Compatibility and experimental status

- Feature and candidate-input changes made on 2026-09-20 (including repaired
  `a_degree`, candidate mass features, charge features, and boundary encoding) make
  all earlier MASShatter checkpoints incompatible for scientific comparison without
  retraining. See the [feature-set audit](../../reference/masshatter_feature_set_audit.md).
- The current default candidate backend is CPU. CUDA construction is an optimization
  path with explicit mode restrictions, not an implicit change in candidate semantics;
  see the [backend guide](../../guides/benchmark_masshatter_cpu.md).
- The connected-fragment beam, shared-across-CE proposals, proposal-only warmup,
  channel hard selection, and alternate intensity/loss formulations are implemented
  experimental controls. Their measured status and next checks belong in the linked
  findings and newer plans, especially the [improvement plan](masshatter_improvement_plan.md),
  [loss review](../../findings/loss_setup_review_2026_09_13.md), and
  [campaign read-out](../../findings/masshatter_campaign_readout_2026_09_21.md).

### Documentation boundary

Sections 1–9 retain the August design rationale, original staged implementation
record, and proposals. Where they conflict with this snapshot or the linked current
references, this snapshot and source win. The project-state entry is the current
cross-document index: [MASShatter state](../../project-state.md#research-navigation).

### Candidate-conditional spectrum correction (2026-08-23)

MASShatter has no fixed fragmentation DAG and therefore does not predict an OOS
class. Its generated ion probabilities are normalized over the selected candidate
set. Observed intensity unmatched by those candidates is reported only through
candidate coverage metrics. Earlier retained/OOS designs documented below are
historical and are superseded by this correction; legacy OOS config keys are ignored.

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
  6.2 specifies a planned template channel for their spectrum-level support.
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

### Shared proposals across collision energies (2026-08-29)

> **Status:** EXPERIMENTAL ABLATION — do not use as the default model.

`mass_shatter_share_proposal_across_ev` reuses one CE-independent proposal for
spectra in the same acquisition group. Its proposal loss is evaluated against a
merged group target, while the primary intensity loss remains CE-specific. This
removes mutually inconsistent per-spectrum supervision from the shared proposal,
but it does not remove bias from the observed CE distribution.

The merged target gives every observed spectrum equal total intensity. Consequently,
densely sampled CE regions contribute more proposal mass than rare CE regions. With
a fixed candidate budget, the shared support is therefore optimized for the empirical
CE distribution and may omit fragments that are important only at uncommon low or
high energies. The current per-spectrum loss interface also fans the group proposal
loss back to every member; with `train_sample_weight: none`, groups containing more CE
spectra receive more total proposal weight. Both effects can make aggregate validation
metrics look acceptable while reducing robustness under a shifted CE distribution.

Keep `mass_shatter_share_proposal_across_ev: False` as the mainline setting and retain
CE conditioning in the proposal heads. If proposal computation must be reduced, prefer
sharing the expensive molecular encoder while applying a small CE-conditioned proposal
head. A CE-balanced merged target—binning CE and weighting occupied bins equally—is a
useful controlled ablation, but it still imposes CE-independent candidate support and
must be evaluated by CE bin and under CE-distribution shift.

### 1.1 Shared dual atom-bond backbone proposal (2026-08-23)

> **Status:** PROPOSED — not implemented or validated. The current
> `GINE_GSA + BondFocusedBreakHead` remains the control.

The representative `a2_b3` configs currently use two different stages:

```
mol_num_layers: 2
  -> two GINE_GSA atom-encoder blocks

mass_shatter_bond_num_layers: 3
  -> three alternating bond -> atom -> bond refinement blocks inside
     BondFocusedBreakHead
```

`mass_shatter_bond_num_layers` therefore does not count bond-only layers. Each
block pools bond states into their endpoint atoms, updates the temporary atom
states, and then updates each bond from its two endpoints. The final bond states
are consumed by the cleavage MLP and then discarded. The temporary atom states
are also local to the head. In particular:

- charge scoring receives the atom encoder output, not the bond-refined atoms;
- fragment and complement embeddings pool the atom encoder output;
- boundary embeddings pool the average of the two endpoint atom embeddings,
  not the explicit refined bond state;
- the intensity transformer therefore has no direct path from the refined bond
  representation.

The proposed change is to promote this temporary atom/bond refinement into a
shared encoder that returns both representations. In the first shared-local arm,
the coupled blocks **stack on top of the unchanged two-layer GINE_GSA atom
encoder**; they do not replace it:

The diagram below is the **target design after the staged sub-arms**, not the
first Arm 3 comparison. That first sub-arm still routes `A_enc` rather than `A3`
to charge and fragment/complement pooling.

```
atom features -> GINE_GSA x2 -> A_enc -> AtomProjection -> A0 --+
                                  |                              |
                                  | endpoint summaries           |
bond features --------------------+-> BondProjection -> B0 ------+
                                                                 |
                                      coupled blocks 1--3 <-------+
                                             |             |
                                             v             v
                                            A3            B3
                                          /    \          /   \
                                    charge   fragment  break  boundary
```

Initialize the streams from the existing atom-backbone output:

```
A_enc = GINE_GSA_2(atom_features, directed_bonds, bond_features)
A0 = AtomProjection(A_enc)

B0 = BondProjection(
    bond_features,
    A_enc[u] + A_enc[v],
    abs(A_enc[u] - A_enc[v]),
)
```

This definition keeps the control's atom encoder fixed and changes only whether
the coupled atom/bond states become reusable outputs. Replacing GINE_GSA with an
interleaved dual encoder is a later arm and must not be folded into this first
comparison.

For each coupled block:

1. pool incident bond states into their endpoint atoms;
2. update atom states with a residual connection and normalization;
3. update each bond from its state and the symmetric endpoint summaries
   `A[u] + A[v]` and `abs(A[u] - A[v])`;
4. optionally apply a graph-scoped global mixer independently to atoms and
   bonds;
5. retain both outputs for the next block.

The symmetric endpoint summaries preserve invariance to the stored orientation
of an undirected chemical bond. Explicit line-graph materialization is not
required to reach the same one-hop bond neighbourhood, but the current endpoint
pool is only **mean-field line-graph-style propagation**, not general line-graph
message passing. A bond receives a degree-normalized mean of incident bond
states, including itself, rather than a learned message from each neighbouring
bond. It therefore cannot represent neighbour-specific bond-pair interactions.
Shared-atom and bond-pair attention biases are proposed later precisely to add
that missing distinction.

The target design keeps acquisition context in the task heads rather than in the
shared structural encoder:

```
break_logit[e] = BreakMLP(B3[e], experiment_context)
charge_logit[v] = ChargeMLP(A3[v], experiment_context)

fragment_embedding   = pool(A3 over fragment atoms)
complement_embedding = pool(A3 over complement atoms)
boundary_embedding   = pool(B3 over cut bonds)
```

This preserves the existing `proposal_selection_only` separation only when
`separate_proposal_encoder: False`: detach both `A3` and `B3` before the proposal
heads while leaving the live states available to the intensity path.

Arm 3 is **not defined for `separate_proposal_encoder: True`**. That option makes
the proposal and intensity encoders fully disjoint. A bond state produced only by
the proposal encoder cannot also provide the live intensity boundary embedding,
so such a run would silently lose the arm's main motivation and retain the
endpoint-average boundary on the intensity side.

If the single shared encoder wins and gradient interference still motivates
separation, test a later explicit **dual-separate** arm: the intensity encoder
gets its own coupled atom/bond stack `(A_int, B_int)`, and the proposal encoder
gets a second disjoint stack `(A_prop, B_prop)`. `B_int` supplies boundary
features; `B_prop` supplies break logits. This roughly doubles the added coupled
backbone cost and must be reported as a distinct architecture, not as the shared
Arm 3. Until that follow-up is justified, separate-proposal configs retain their
current endpoint-average intensity boundaries and are excluded from the Arm 3
comparison.

Removing context from `B0` is a behavioral change, not a free refactor: the
current `BondFocusedBreakHead.bond_project` consumes acquisition context before
all three refinement blocks. The shared-local experiment must include a matched
arm that retains context in `B0`; only a measured context-in-backbone versus
context-in-head comparison can justify the molecule-only target design.

Sharing also changes gradient routing under `proposal_selection_only`. Today the
temporary bond stream is trained through the proposal objective (weight `0.1` in
the representative proposal-selection configs), while the intensity path never
uses it. Once live `B3` supplies boundary embeddings, the spectrum intensity loss
also trains the bond representation and may dominate the smaller proposal term.
This is a substantive multi-objective optimization change. The first shared
experiment must therefore include:

- a matched `mass_shatter_proposal_loss_weight: 0.1` arm;
- a stop-gradient boundary arm using `B3.detach()` in the intensity path, so the
  proposal-only and joint-gradient representations can be compared at identical
  forward features;
- gradient-norm or loss-scale reporting for the proposal and intensity paths;
- a proposal-loss reweighting control if joint training degrades bond ranking.

Do not attribute a null shared-encoder result to the representation until this
gradient-routing control has ruled out proposal-signal dilution.

The direct bond-state boundary representation is the clearest downstream reason
to share the encoder. A fragment boundary is defined by the cut bonds, whereas
the current endpoint average can represent the adjacent atoms but cannot retain
an independently refined state for each broken bond.

#### 1.1.1 Global mixer: EFA is an ablation, not the default choice

The current EFA implementation uses normalized `ELU(x) + 1` kernel attention.
For `N` tokens, `H` heads, and head dimension `D`, its leading attention cost is:

```
EFA:              O(N * H * D^2)
softmax attention: O(N^2 * H * D)
```

With the common width 384 and four heads, `D = 96`; the arithmetic crossover is
therefore around 96 tokens per branch. This crossover is outside nearly the
entire NIST20 distribution. A full-table measurement of all 35,025 rows in
`data/proc/nist20/mol_df.pkl` gives the following molecular-graph token counts
(heavy atoms only; implicit hydrogens are not graph nodes):

| tokens per molecule | Median | p90 | p99 | Max | Fraction above 96 |
|---|---:|---:|---:|---:|---:|
| atoms | 20 | 44 | 82 | 186 | 0.445% |
| bonds | 21 | 47 | 87 | 195 | 0.725% |

At the median, the leading EFA contraction is about 4.6--4.8 times the leading
exact-attention contraction (`D/N`), before accounting for kernel quality. EFA's
asymptotic FLOP advantage is therefore not realized for more than 99% of either
branch. Actual latency must still be profiled because optimized kernels and
padding affect wall-clock performance, but the token distribution puts the
burden of proof on EFA rather than softmax.

The current EFA implementation also has a first-order transient-memory cost. It
materializes both `KV_nodes` and `KV_node` with shape `(N, H, D, D)`. For a
representative batch of about 3,000 atom tokens at `H = 4`, `D = 96`, each tensor
is about 422 MiB in FP32, so the pair reaches about 844 MiB during one EFA call.
FP32 is the operative estimate for the control configs summarized here; BF16 is
kept as a separate, explicitly tagged precision ablation rather than assumed.
Non-reentrant activation checkpointing is unconditional in the EFA path and
prevents all layer intermediates from simply accumulating for backward, but the
same large allocation is still required transiently in the forward call and its
backward recomputation. A bond EFA branch repeats approximately the same work and
allocation because the bond count closely tracks the atom count; its exact peak
depends on whether the branches execute sequentially. By comparison, per-graph
softmax attention padded near 90--96 tokens is only on the order of tens of MiB
for 128 graphs before other attention intermediates. Measure allocated and
reserved CUDA peaks for the complete model rather than estimating total model
memory from one tensor alone.

More importantly, kernel factorization does not naturally admit an arbitrary
learned bias for every token pair without giving up its linear form. That makes
the current EFA a weak match for the main representational idea in Graphormer and
the Dual Graph Transformer: put molecular topology and structure directly into
the attention score. Useful atom-pair and bond-pair biases include:

- shortest-path distance in the atom or bond graph;
- multi-hop bond or intervening-atom features;
- direct incidence or shared-atom relation;
- ring membership and relative ring position;
- optional geometric distance or angle encodings when reliable coordinates are
  available.

The first global dual model should therefore use structurally biased softmax
attention. EFA remains a required matched ablation and a scalability fallback if
the measured token distribution or memory profile justifies it. This follows the
local-message-passing + structural-encoding + global-attention decomposition in
[GraphGPS](https://openreview.net/pdf?id=lMMaNf6oxKM), while the pairwise bias
choice follows [Graphormer](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html)
and the molecular dual representation follows the
[Dual Graph Transformer](https://www.nature.com/articles/s41467-026-75005-9).

The atom-side structural-softmax implementation is **already present** and must
be reused rather than rebuilt. `Graphormer` in
`src/fragnnet/model/nn_blocks.py` supplies
degree, shortest-path, and up-to-five-hop edge-path attention biases. MASShatter
already has merged and MA/MI-unmerged `graphormer512x5` unique-fragment configs.
Those configs use an atom-only Graphormer with
`mass_shatter_bond_num_layers: 0`; they establish an existing performance
reference, although their width 512, depth 5, and endpoint head are not a
parameter-matched dual-backbone ablation. The genuinely missing components are:

- explicit reusable bond-stream outputs;
- bond-graph structural attention and its pairwise biases;
- controlled atom/bond cross-stream fusion;
- a matched-width/depth comparison against the existing atom Graphormer.

A bounded first structural-attention design is:

```
block 1: local atom <-> bond update
block 2: local update + biased atom attention + biased bond attention
block 3: local update + biased atom attention + biased bond attention
```

Global attention in every block is not a requirement. The local-only first block
retains a chemical neighbourhood prior and limits cost before global mixing.

#### 1.1.2 Ordered implementation and ablation sequence

Do not implement the complete dual transformer first. Advance in the following
order, keeping candidate enumeration, losses, context, data splits, and realized
candidate budgets fixed:

1. **Current control:** `GINE_GSA + BondFocusedBreakHead`; endpoint-average
   boundary features.
2. **Refactor-equivalence check:** extract the current refinement blocks so they
   return `A3, B3`, but preserve the current logits and do not consume those
   states downstream. Require numerical parity before training a changed model.
3. **Shared-local dual encoder:** stack those blocks on the unchanged two-layer
   GINE_GSA output, use a small context-conditioned cleavage MLP, and replace the
   endpoint-average boundary feature with `B3`. Run the context-location and
   stop-gradient/proposal-weight controls defined above. In the first sub-arm,
   keep charge and fragment/complement pooling on the original GINE_GSA atom
   output so `B3` boundary reuse is the only downstream representation change.
   Promote `A3` to those consumers only in a second sub-arm. Add no global mixer.
   This arm requires `separate_proposal_encoder: False`.
4. **Existing atom-Graphormer reference:** evaluate the already configured
   `graphormer512x5` endpoint-head model, then construct a matched-width/depth
   atom-Graphormer control. Do not attribute its atom-side structural biases to a
   newly implemented dual model.
5. **Bond-EFA arm:** add graph-scoped EFA over `B` after each selected coupled
   block while leaving the atom side fixed.
6. **Unbiased bond-softmax arm:** replace bond EFA with ordinary graph-scoped
   softmax attention without bond-pair structural bias, again leaving the atom
   side fixed.
7. **Structural dual-softmax arm:** reuse the existing Graphormer atom branch and
   add the missing shortest-path, shared-atom, multi-hop, and ring biases to the
   bond attention scores.
8. **Fully interleaved dual model:** move local atom GINE, local bond updates,
   global atom attention, global bond attention, and cross-stream fusion into the
   same repeated block only if the preceding structural-softmax arm wins.
9. **Dual-separate follow-up, only if needed:** after a shared encoder wins, give
   the intensity and proposal paths disjoint coupled atom/bond stacks and compare
   against the matching separate-encoder control at reported parameter count and
   memory. Do not include this cost-doubled design in the initial Arm 3 test.

Arms 5--7 answer separate questions: whether global bond communication helps,
whether exact softmax improves on kernel attention, and whether any softmax gain
actually comes from molecular structural bias. Arms 2--3 prevent atom-backbone,
context-location, and gradient-routing changes from being hidden inside that
comparison.

#### 1.1.3 Evaluation and stop gates

Compare every arm at both matched hidden width and reported total parameter
count; these answer different practical questions. Report:

- final held-out spectrum similarity under the standard instrument-tolerance
  evaluation;
- bond ranking and productive-mask recall at matched selected-mask budgets;
- unique-fragment candidate coverage and oracle spectrum score;
- base-peak accuracy and matched-peak ranking diagnostics;
- training throughput, inference latency, and peak GPU memory;
- results by molecule atom count, bond count, adduct, collision energy, and
  fragmentation mode.

Use the same three seeds and paired evaluation contract as the canonical model
improvement plan. A one-seed run may reject a clear failure but may not promote
the backbone.

Stop the line when the shared-local encoder fails to improve either bond/mask
ranking or final spectrum quality: global attention should not be added merely
because it is architecturally appealing. Prefer EFA over structural softmax only
when it preserves the validated accuracy gain and materially improves measured
throughput or memory on the actual molecule-size distribution.

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

### 5.2 TODO — H-transfer range ablation

The current production range is `delta_H in [-4, +4]`, which gives 18 CRF/CMF
channels per fragment. Run a matched ablation with
`mass_shatter_max_h_transfer` in `{4, 6, 7}`:

- `4`: 18 channels (`-4` through `+4`, crossed with CRF/CMF)
- `6`: 26 channels (`-6` through `+6`, crossed with CRF/CMF)
- `7`: 30 channels (`-7` through `+7`, crossed with CRF/CMF)

Hold the architecture, candidate budget, loss weights, and training schedule
fixed. Compare instrument-tolerance cosine/JSS, candidate recall/wRecall and
precision, active-channel count, probability/intensity by `abs(delta_H)`, and
runtime/memory. Do not select the range using 0.1 Da-binned metrics. Adopt the
smallest range that improves held-out instrument-tolerance cosine or wRecall
without degrading precision or calibration. Treat GLACIER's published `-6`
through `+6` range as action-space context, not evidence that MASShatter needs
the same range.

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

### 6.1 Unique-fragment pathway marginalization — IMPLEMENTED 2026-08-19

The original pathway-token implementation sent every component occurrence from
every retained cut mask through the intensity transformer. This is probabilistically
valid but computationally wasteful: many masks produce the same precursor atom set
through spectator cuts, redundant ring openings, or cuts confined to the neutral
complement. Those occurrences must contribute probability, but they do not need
separate learned fragment representations.

Unique-fragment mode separates **latent component pathways** from **fragment
tokens**:

1. Enumerate the probability-ranked cut masks and their connected components.
2. Assign every component occurrence a unique ID determined only by its precursor
   atom set. Record the first occurrence as its canonical representative.
3. Pool atom charge-affinity logits once per unique atom set, broadcast that score
   back to every component occurrence, and normalize charge retention among the
   sibling components of each mask.
4. Marginalize all charged occurrences of the same atom set:

   ```
   log P(F)
     = logsumexp_{C: F in components(G \ C)}
         [log P(C) + log P_charge(F | C, precursor_type)]
   ```

5. Run component/complement/boundary pooling and the intensity transformer once per
   unique atom set. The joint ion-mode/H-transfer probabilities are conditioned on
   this marginalized fragment probability.

The charge score can be reused because it is computed from the fragment's atoms and
their context-conditioned embeddings; only its normalization changes with the
sibling components present in a particular mask. The normalization therefore still
models competition for charge within each cleavage event, while `logsumexp` preserves
the probability contributed by every retained pathway.

The canonical cut count of a unique fragment is the number of precursor bonds that
cross its atom-set boundary. It is not the total cut count of an arbitrary parent
mask: spectator cuts in the neutral complement must not make a simple fragment look
like a high-cut fragment.

This is not a direct atom-mask or fixed-query fragment decoder. Fragment identities
remain chemically derived from latent bond-cleavage events, there is no Hungarian
matching or fragment-label supervision, arbitrary cut cardinalities remain possible,
and the unenumerated Bernoulli mask mass remains the explicit OOS probability. Mask
selection is still detached top-K, but the spectrum loss backpropagates through the
live probabilities of every selected mask, the per-mask charge normalization, and
the unique-fragment ion head.

Configuration:

- `mass_shatter_unique_fragment_mode: True` uses
  `mass_shatter_unique_fragment_target` as the requested unique-fragment token
  target. The final complete mask can add several
  new atom sets, so the realized token count can slightly exceed the target.
- `mass_shatter_ranked_mask_budget` bounds learned latent-mask enumeration. Empty
  and single-cut masks retain their established outside-budget support semantics.
- `mass_shatter_selected_mask_limit` is the hard selected-mask limit when unique
  fragment mode is disabled.
- The first controlled experiments keep both unique-mode values at 512. Increasing
  latent masks without a proportional gain in unique fragments is deliberately avoided.
- `mass_shatter_use_mask_residuals: False` gives the cleanest probabilistic ablation:
  the bond/charge model fixes fragment mass and the transformer only allocates ion/H
  channels.

The implementation logs count-based utilization (`productive_mask_fraction`,
`duplicate_component_pathway_fraction`, and
`latent_pathways_per_unique_fragment`), target status
(`unique_fragment_target_fill_fraction` and `unique_fragment_target_reached`), and
probability-aware concentration (`unique_fragment_effective_count`,
`unique_fragment_top1_mass_fraction`, and
`duplicate_pathway_probability_fraction`). The last metric is the fraction of
retained fragment probability contributed beyond the single best pathway for each
atom set; unlike duplicate counts, it shows whether marginalizing redundant pathways
materially changes the predicted distribution. These metrics distinguish a support
shortage from a redundancy problem and must be reported with runtime and candidate
weighted recall.

Reference configurations:

- `config/massshatter/ma_mi_unmerged/unique_fragment/nist20_ma_mi_unmerged_unique512_gine_gsa384_bondcentric_a2_b3_transformer_full_context_no_mask_residual.yml`
- `config/massshatter/ma_mi_unmerged/unique_fragment/nist20_ma_mi_unmerged_unique512_gine_gsa384_bondcentric_a2_b3_transformer_full_context_mask_residual.yml`
- `config/massshatter/merged/unique_fragment/nist20_merged_unique512_gine_gsa384_bondcentric_a2_b3_transformer_ce_embed_no_mask_residual.yml`
- `config/massshatter/merged/unique_fragment/nist20_merged_unique512_gine_gsa384_bondcentric_a2_b3_transformer_ce_embed_mask_residual.yml`

### 6.1.1 Mask-residual shortcut and proposal-recall risk

The mask residual is not only a provenance or interpretability concern. The bond
logits have two jobs: they assign base probability to retained fragments, and they
choose which masks enter the deterministic top-K candidate set. A transformer can
improve the first job while starving the second of useful supervision.

Candidate selection is discrete. `top_k_cut_masks` ranks masks from detached bond
logits, and with `mass_shatter_explore_masks: 0` the same learned ranking
deterministically chooses the enumerated set. The selected masks' probabilities are
then recomputed from the live logits, so gradients can reach the bond head through
their scores, but no gradient passes through the top-K choice itself. A fragment
whose generating mask is not selected is absent from the transformer's input and
cannot be recovered by any downstream residual.

For retained fragment base log probabilities `l_i` and learned residuals `r_i`, the
implemented correction is:

```
c_i = l_i + r_i
q_i = c_i - logsumexp_j(c_j) + logsumexp_j(l_j)
```

The final term is important. It guarantees
`sum_i exp(q_i) = sum_i exp(l_i)`, so the transformer may redistribute the shape of
the retained distribution but cannot change its total probability mass. The base
model therefore continues to receive supervision for the retained versus
out-of-support split. Removing this normalization would let the residual absorb
that split as well.

The remaining shortcut is that the intensity transformer receives `l_i` as an
input feature. It can learn `r_i` approximately equal to a desired score minus
`l_i`, making the normalized shape nearly independent of the bond model. The main
spectrum loss can then fall while the bond head receives mostly the aggregate
retained/OOS signal instead of per-peak relative-ranking gradients. This creates a
feedback loop:

```
residual corrects the retained distribution
  -> spectrum CE falls
  -> bond-head shape supervision weakens
  -> a weak bond proposer chooses the next top-K set
  -> omitted fragments remain unreachable by the residual
```

The result is a proposal-recall ceiling hidden by better within-set scoring. It also
makes bond logits less useful as calibrated breakability artifacts and weakens the
structural regularization supplied by the independent-bond model.

Measure this before changing the model. At evaluation, compare oracle coverage
between the matched `mask_residual` and `no_mask_residual` runs at the same learned
mask limit, unique-fragment target, and realized candidate count: the fraction of
observed peaks compatible with at least one ion generated by any enumerated mask,
ignoring predicted intensity. Coverage is fixed before the residual is applied.
Interpret the comparison as follows:

- Better spectrum scoring with equal or worse oracle coverage in the residual run
  supports the shortcut hypothesis: scoring improved without improving the
  generator that sets the support ceiling.
- Equal and high coverage makes this mainly a provenance and regularization issue,
  and therefore lower priority.
- Low coverage in both runs establishes a candidate-generation bottleneck but does
  not by itself show that the residual caused it.

If the first outcome is observed, keep the comparative transformer and make two
surgical changes:

1. Remove the base mask/fragment log probability from the transformer's input
   features. Structural, complement, boundary, size, cut-count, and experiment
   context features remain, so this closes the direct cancellation route without
   removing comparative scoring capacity.
2. Add an auxiliary base-only sparse cross-entropy term, computed from the
   pre-residual fragment distribution with the same retained/OOS accounting:

   ```
   L_total = L_corrected + lambda_base * L_base_only
   ```

   Start with `lambda_base` in the `0.1`--`0.3` range. This requires the bond/charge
   path to remain a competent standalone predictor and restores per-peak gradient
   to the logits that rank future top-K masks. It cannot directly supervise a
   correct mask that is currently absent, but improved ranking of reachable
   explanations can change subsequent top-K sets.

A residual-magnitude penalty is a weaker fallback: it suppresses the shortcut by
restricting useful corrections and may exchange accuracy for structure. The
base-only auxiliary loss instead preserves the corrected predictor's freedom while
independently supervising the proposal model.

#### Defect 1 implementation: repair the residual factorization

MASShatter remains one jointly trained end-to-end model. Separating formation from
detected intensity means giving the existing heads distinct responsibilities and
losses inside the same forward/backward pass; it does not mean training a generator,
materializing its outputs, and training a second intensity model afterward.

The current retained-shape calculation is:

```
log P_final(i)
  = log R + log_softmax_i(l_i + r_i)
```

where `l_i` is the base fragment log probability from the bond/charge path, `r_i`
is the intensity-transformer residual, and `R` is the base model's total retained
mass. This factorization has a useful inductive bias: formation probability remains
a structural prior and the transformer learns a comparative correction. The problem
is not the addition itself; it is that the transformer receives `l_i` directly and
the base path has no standalone shape objective.

The primary change is therefore the minimal repaired model:

```
r_i = IntensityTransformer(structure_i, complement_i, boundary_i, context)
log P_final(i) = log R + log_softmax_i(l_i + r_i)
L = L_final + lambda_base * L_base_only
```

Use one shared precursor encoder, one online candidate-generation pass, one combined
loss, and one optimizer. Remove `l_i` only from the transformer's input features;
do not remove it from the final base-plus-residual score. This closes the direct
cancellation route while preserving the independent-bond prior in the final
prediction. The auxiliary loss prevents the residual from starving the proposer
even if other structural features let it infer part of the base ranking.
Top-K membership remains discrete and detached, so this is jointly trained
end-to-end but not differentiable through the candidate identities themselves.

A fully separated conditional-intensity variant is a stronger ablation:

```
log P_separated(i) = log R + log_softmax_i(s_i)
```

It is cleanly identifiable, but it discards relative formation probability after
enumeration and gives the transformer a free within-support allocation. That trades
away a structural regularizer and is not the default implementation path.

Run the matched comparison in this order:

1. **A — current control:** base probability is an input feature and the final score
   is `l_i + r_i`, with no base-only CE.
2. **B — primary repaired model:** remove the probability input, retain `l_i + r_i`,
   and add base-only CE. The initial fixed-weight range `0.1`--`0.3` has now shown
   a dose-dependent final-metric regression; use the revised experiment below.
3. **C — full-separation ablation:** use absolute structural intensity logits `s_i`
   for retained shape, while keeping the same retained/OOS mass and base-only loss.

Keep mask limits, realized candidate budgets, transformer capacity, optimizer,
training schedule, and seeds matched. Compare final cosine/JSS, base-only spectrum
metrics, deterministic candidate weighted recall, retained/OOS calibration, bond
logit sign and ranking diagnostics, and gradient magnitude reaching the bond head.
Advance B as the Defect 1 arm if it improves the base predictor and proposer
diagnostics without a reproducible final-metric regression; do not treat it as the
complete architectural repair. Retain C only as an ablation if it beats B on held-out
final metrics without reducing candidate coverage, base-only competence, or OOS
calibration. The combined target in Section 6.1.2 remains the adoption candidate.

#### Proposal-selection implementation (corrected 2026-08-23)

The opt-in `mass_shatter_proposal_selection_only` path separates candidate retrieval
from conditional intensity without changing candidate generation. Bond-mask ranking,
charge scoring, unique-fragment deduplication, and the ranked-mask budget remain
unchanged. After selection, the candidate transformer owns the distribution over the
generated candidates:

```
log P_final(i | selected candidates) = log_softmax_i(s_i)
```

Proposal probabilities therefore affect candidate membership but receive no primary
spectrum-loss gradient. A separate intensity-weighted support loss trains the
bond/charge proposer. For each true peak, it sums the normalized base probability of
every unique fragment with at least one valid ion channel inside the instrument
tolerance. Each fragment is counted once per peak, regardless of how many CRF/CMF or
`delta_H` channels match, and the loss uses no learned ion/H probability. Proposal
inputs are detached from the shared precursor and context encoders, so proposal
gradients update only the bond and charge heads; the final spectrum trains the shared
precursor encoder and candidate transformer.

The earlier learned-OOS experiment is retired. MASShatter has no fixed fragmentation
DAG and no identifiable OOS class: unenumerated mask probability is not unexplained
spectral intensity, and multiple masks can produce the same fragment. The former V2
configs were therefore removed rather than retained as meaningful ablations.

Reference configurations:

- merged U256/E1024: `config/massshatter/merged/unique_fragment/nist20_merged_unique256_enum1024_gine_gsa384_bondcentric_a2_b3_transformer_ce_embed_proposal_selection_only.yml`
- unmerged U256/E4096: `config/massshatter/ma_mi_unmerged/unique_fragment/nist20_ma_mi_unmerged_unique256_enum4096_gine_gsa384_bondcentric_a2_b3_transformer_full_context_proposal_selection_only_inst_tol.yml`

#### Separate proposal encoder ablation (2026-08-23)

This ablation tests whether the detached intensity-GNN representation is misaligned
with proposal ranking. With `mass_shatter_separate_proposal_encoder`, the bond and
charge heads consume an independent GNN representation:

```
Proposal GNN  -> bond/charge heads -> candidate support -> L_proposal
Intensity GNN -> candidate transformer + ion/H          -> L_spectrum
```

The acquisition context remains shared but detached on the proposal path. Candidate
generation and the H-independent proposal-support loss are identical to the
single-encoder control. Thus proposal loss updates the proposal GNN and bond/charge
heads, while primary spectrum loss updates the intensity GNN and candidate
transformer. Each comparison changes only
`mass_shatter_separate_proposal_encoder`, apart from run names, tags, and comments.

Matched experiment pairs:

- merged control: `config/massshatter/merged/unique_fragment/nist20_merged_unique256_enum1024_gine_gsa384_bondcentric_a2_b3_transformer_ce_embed_proposal_selection_only.yml`
- merged separate encoder: `config/massshatter/merged/unique_fragment/nist20_merged_unique256_enum1024_gine_gsa384_bondcentric_a2_b3_transformer_ce_embed_proposal_selection_only_separate_proposal_encoder.yml`
- unmerged control: `config/massshatter/ma_mi_unmerged/unique_fragment/nist20_ma_mi_unmerged_unique256_enum4096_gine_gsa384_bondcentric_a2_b3_transformer_full_context_proposal_selection_only_inst_tol.yml`
- unmerged separate encoder: `config/massshatter/ma_mi_unmerged/unique_fragment/nist20_ma_mi_unmerged_unique256_enum4096_gine_gsa384_bondcentric_a2_b3_transformer_full_context_proposal_selection_only_separate_proposal_encoder_inst_tol.yml`

#### Measured finding: fixed base-only CE trades final cosine for support (2026-08-20)

Three matched merged-spectrum grid sweeps varied
`mass_shatter_base_ce_weight` over `0.1`, `0.2`, and `0.3` for unique-fragment
targets 128, 256, and 512:

- W&B sweep `o6gsdyvv`: U128, ranked-mask budgets 256 and 2048;
- W&B sweep `9pmv90u1`: U256, ranked-mask budgets 256 and 2048;
- W&B sweep `4wdysag2`: U512, ranked-mask budgets 512 and 2048.

At the lower mask budget, mean best validation cosine decreased monotonically:

| Fixed base-CE weight | Mean best validation cosine |
| -------------------: | --------------------------: |
|                  0.1 |                     0.72846 |
|                  0.2 |                     0.72626 |
|                  0.3 |                     0.72231 |

The `0.3` mean contains two completed runs and the U512 run at epoch 121. At the
2048-mask budget, the mature `0.1` and `0.2` arms showed the same ordering:
`0.73240` versus `0.73044`. Those six arms crashed late at epochs 117--176, so the
comparison uses their best checkpoint metric rather than successful completion. The
2048-mask `0.3` arms were still too immature to compare.

This is not a pure support regression. At mask budget 2048, increasing the weight
from `0.1` to `0.2` raised mean candidate recall from `0.7064` to `0.7156` and mean
candidate weighted recall from `0.8927` to `0.9007`, while mean best cosine fell by
`0.00196`. The auxiliary objective therefore improves candidate inclusion but harms
the final probability allocation. The sweeps have no `0.0` arm, and a matched
proposal-anchored zero-weight run was not found; they establish that more weight is
harmful over `0.1`--`0.3`, not that disabling the auxiliary term is optimal.

The most plausible mechanism is gradient interference, because the implemented
base-only loss is not proposer-only. `base_pred_logprobs` combines the base
fragment probabilities with the same `ion_h_log_probabilities` produced by the
candidate intensity transformer for the final spectrum. Its gradient therefore
updates the shared transformer trunk and ion/H head as well as the bond, charge,
and precursor paths. Meanwhile, proposal anchoring removes the base probabilities
from the transformer's input even though the final score remains `l_i + r_i`, so
the residual must correct a moving base offset that it cannot observe directly.
Consistent with this interpretation, the U128/256 validation base-to-final CE gain
shrinks from `0.3666` at weight `0.1` to `0.3226` at `0.2` and `0.2972` at `0.3`.
The raw weighted base-loss magnitude also grows from roughly 8% of the primary-loss
magnitude at `0.1` to 25% at `0.3`; this is not a gradient-norm estimate, but it
shows that the nominally auxiliary term is not negligible.

Revise the Defect 1 experiments in this order:

1. **Isolate the shared-head gradient.** In the base-spectrum branch only, use
   `ion_h_log_probabilities.detach()`. This preserves the base-CE value and its
   gradients to the bond/charge proposal while preventing it from training the
   candidate transformer through the shared ion/H prediction. Compare fixed `0.1`
   with and without the detach at matched seed and budget. Advance the change only
   if the candidate recall gain remains and final cosine recovers.
2. **Measure gradient conflict directly.** On matched batches, log gradient norms
   and cosine similarity between `L_final` and `L_base_only` for the bond head,
   charge head, precursor encoder, candidate-transformer trunk, residual head, and
   ion/H head. The working hypothesis is alignment in the proposer and conflict in
   shared intensity parameters; do not redesign the shared encoder without this
   attribution.
3. **Reduce late-training pressure.** Compare fixed `0.1` with a warm-up schedule
   that starts at `0.1` and decays to a small positive floor. Proposal-anchored setup
   currently rejects an exact zero weight, so an exact-zero endpoint requires an
   explicit compatibility change rather than a silent config edit.
4. **Search below the observed boundary.** Test `0.01`, `0.025`, `0.05`, and `0.1`
   after the shared-head ablation. Include an exact `0.0` control only after defining
   how the anchored proposer remains supervised. The present evidence makes `0.2`
   and `0.3` poor default choices.
5. **Control support through selection.** Compare ranked-mask budget and a calibrated
   minimum mask-probability threshold at matched realized candidate count. Candidate
   support is a selection problem; do not use a stronger full-spectrum auxiliary loss
   as the primary recall knob if budget or thresholding can recover the same support
   without disturbing intensity gradients.
6. **Escalate to a fully proposer-local objective only if needed.** If detaching ion/H
   is insufficient, block base-loss gradients into the shared precursor encoder and
   train only the bond/charge proposal parameters, or replace full-spectrum base CE
   with an explicit candidate-ranking objective. This is a larger change and needs
   its own matched control because it changes which representation learns proposal
   information.

### 6.1.2 Independent-mask downward closure and correlated-cut support

**Implementation priority:** treat this as the second half of the same architectural
repair, not as a contingency that Variant B can make unnecessary. Variant B restores
supervision to the proposal distribution, but no loss can make an independent
Bernoulli distribution express positive cleavage synergy. Conversely, a correlated
proposal does not stop the residual from cancelling proposal scores. Evaluate the two
changes separately for attribution, then combine them for the repaired model.

The residual shortcut composes with a separate expressivity constraint in the
independent Bernoulli mask model. For two cut sets `T` and `S` with `T` a strict
subset of `S`:

```
log P(S) - log P(T) = sum_{b in S ∖ T} logit_b
```

If every bond logit for a molecule is strictly negative, the MAP mask is empty and
every strict subset has higher probability than its supersets. The exact top-K
mask prefix is then a **downward-closed family**: selecting a `d`-cut mask requires
all `2**d - 1` proper subsets to appear earlier. Zero logits introduce ties, and
positive logits invalidate the cut-set downward-closure guarantee. More generally,
the heap is always downward-closed in flip sets around its possibly nonempty MAP
mask.

The current model does not constrain logits to be negative, so the empty-MAP
condition is a hypothesis about the trained fragmentation prior, not an identity of
the implementation. Test it directly by logging the positive-logit fraction and MAP
cut count per precursor. Training-time exploration can also add masks outside the
exact prefix, but evaluation with `mass_shatter_explore_masks: 0` uses the
deterministic ranking and exposes this constraint directly.

#### Consequences when the MAP mask is empty

Independence cannot express positive cleavage synergy. Holding every other bond
state fixed, for two bonds `i` and `j`:

```
P(R ∪ {i, j}) / P(R ∪ {i}) = exp(logit_j)
```

When `logit_j < 0`, the pair must rank below the corresponding singleton. The
model therefore cannot represent a mode in which two bonds are favored together
while either isolated cut is disfavored. It can rank the pair above a singleton
only by making the other bond individually more likely cut than intact. The
independent model has unit pairwise odds ratio and cannot represent the underlying
positive correlation itself.

Ring cleavage makes the budget cost concrete. Deleting one non-bridge ring edge
leaves the molecular graph connected, so atom-set-based unique-fragment mode maps
that pathway to the same full-precursor atom set as the empty mask. It may still
contribute probability to a precursor-mass ion, but it creates no new fragment
support. A correlated two-edge ring release cannot enter an empty-MAP top-K prefix
without both support-redundant singleton ancestors consuming earlier ranks.

Depth is also combinatorially expensive. With 25 equal negative logits, ranking is
by cut count, so a budget of 512 contains the empty mask, all 25 singletons, all 300
pairs, and only 186 of the 2,300 triples; it contains no four-cut mask. This is an
illustration, not a general bound on the realized cut histogram. Heterogeneous
near-zero logits can admit selected high-order masks, overlapping high-order masks
can share ancestors, and a downward-closed family of size 512 can contain a mask
with as many as nine cuts if all of its subsets occupy the family. The actual run,
not the equal-logit example, must settle whether depth is excluded in practice.

The existing four-cut SMARTS rules for xanthone, furanocoumarin/psoralen,
morphinan, and anthraquinone establish that useful compound classes contain
correlated high-order products missed by the fixed D3 process. They motivate this
test but do not prove that a trained MASShatter top-K prefix misses the same
products.

This is the support half of the residual failure mode:

```
independent ranking limits which correlated masks reach the candidate set
  -> the residual repairs scores only within that set
  -> final spectrum loss understates the support ceiling
  -> weakened base-shape gradients make the proposer less likely to escape it
```

The two remedies are complementary. Base-only auxiliary CE cannot make an
independent distribution express correlated cuts. A correlated proposal model does
not by itself prevent a downstream residual from cancelling its selected-mask
scores. The repaired target therefore combines a richer cut-set distribution with
the anti-cancellation changes in Section 6.1.1; the remaining gate determines adoption,
not whether the two defects can substitute for one another.

#### Combined target design: accountable grouped proposal plus residual scorer

The target architecture fixes both failure modes in one jointly trained model. Partition
the precursor bonds into free bonds `F` and fused ring systems `G`. Free bonds retain
Bernoulli factors; each ring system emits exactly one categorical choice from its
precomputed admissible cut patterns plus `uncut`:

```
P_base(S | x)
  = product_{b in F} Bernoulli(1[b in S]; p_b)
    * product_{g in G} Categorical(choice_g(S); pi_g)
```

This distribution is exactly normalized because every bond belongs to one group and
every group is normalized. Its exact factored top-K is not downward-closed in individual
bond cuts: a multi-bond ring option is one categorical choice and may outrank all of its
single-bond subsets. The selected masks still pass through the existing charge and
duplicate-pathway marginalization to produce base fragment log probabilities `l_i` and
retained mass `R`.

The intensity transformer receives structure, complement, boundary, size, cut-count,
and experiment context, but receives neither mask nor charge/fragment proposal log
probabilities. It emits comparative residuals `r_i`:

```
log P_base_fragment(i) = log R + log_softmax_i(l_i)
log P_final(i)         = log R + log_softmax_i(l_i + r_i)
P(OOS)                 = 1 - R

L = L_final + lambda_base * L_base_only
```

`L_base_only` uses the same peak matching and retained/OOS accounting as `L_final`.
It makes the grouped bond/option and charge proposal a competent standalone spectrum
predictor, while `L_final` preserves the useful base-plus-residual structural prior.
Top-K identities remain discrete, but selected-mask probabilities are recomputed from
live logits, so both losses update the proposal scores that determine later top-K sets.

The ablation sequence should isolate and then combine the repairs:

1. Independent proposal with the original probability-aware residual (current control).
2. Independent proposal with proposal anchoring and base-only CE (Defect 1 only).
3. Grouped proposal with the original residual interface (Defect 2 only; diagnostic).
4. **Combined target:** grouped proposal, proposal-anchored residual, and base-only CE.

Compare all arms at matched realized mask and unique-fragment budgets. Adoption requires
the combined target to improve correlated-cut candidate coverage and base-only competence
without regressing final held-out spectrum metrics or retained/OOS calibration. The
ring-system grouping, option head, and reference factored top-K implementation exist and
are tested, but are not yet wired into MASShatter candidate generation; therefore the
live model currently implements arm 2, not arm 4.

#### Pre-implementation decision gate

Use the existing DAG only as a reference explanation set, not as ground-truth
fragment labels. For every observed peak compatible with at least one DAG atom set,
compute the minimum precursor boundary-cut count among all compatible explanations.
Report both peak-count and observed-intensity histograms, including ambiguity and
ties, and compare them with what the deterministic learned-mask enumeration covers.
Classify atom sets that cannot be a connected component after precursor-bond
deletion separately; those test the template/new-bond action space rather than
independent versus correlated cut scoring.

- If at least 90% of DAG-explainable observed intensity has a minimum explanation
  at two or fewer cuts and the current enumeration covers it, treat independence as
  provisionally survivable and prioritize the residual/base-only-loss experiment.
- If a material intensity share requires three or more cuts but the learned prefix
  rarely enumerates them, treat cut-set expressivity as the binding constraint and
  run the mixture probe below before designing the autoregressive model.

The 90% threshold is an experiment gate, not proof that independent masks are
adequate. Report sensitivity around it and stratify by scaffold and collision
energy so a strong global average does not hide affected chemical classes.

Current runs already log `mean_mask_cut_count`, `mask_cut_fraction_{0,1,2,3,4plus}`,
candidate coverage by cut-count bucket, and `productive_mask_fraction`. Add the
positive-logit/MAP diagnostics above and an exact fraction of enumerated masks that
produce only the full-precursor component. The latter is the direct ring-opening
budget diagnostic; `1 - productive_mask_fraction` is broader because it also
includes pathways whose fragments were first introduced by another mask.

#### Normalization constraint for replacements

The OOS bucket requires a proper probability distribution over complete cut sets:

```
retained_mass = sum_{S in unique enumerated masks} P(S)
P(OOS) = 1 - retained_mass
```

Any replacement must provide an exact normalized `P(S)` for every selected set and
deduplicate masks before summing them. This makes retained and OOS mass
probabilistically well-defined; empirical calibration remains a separate
measurement.

#### Cheap probe: mixture of independent Bernoullis

Use `M` latent reaction archetypes, initially `M = 8`--`16`:

```
P(S) = sum_z pi_z * product_b p_b(z)**x_b * (1 - p_b(z))**(1 - x_b)
```

Softmax-normalized mixture weights and normalized Bernoulli components preserve a
proper cut-set distribution. The mixture can represent low global marginal break
rates together with a high-probability correlated cut set by assigning that set to
a low-weight archetype whose coordinated bonds have positive component logits. If
every component keeps every logit negative, their weighted sum remains
downward-closed, so component-specific positive logits are essential to the probe.

Reuse `top_k_cut_masks` inside each component, take a deduplicated union, and rescore
every union mask under the complete mixture. This is a useful overcomplete candidate
probe, but a fixed top-K union from each component is **not guaranteed** to contain
the mixture's global top-K. Treat it as approximate search unless enumeration is
continued with an upper-bound certificate. Regardless of search exactness, the
retained mass of the final deduplicated set is exact after full mixture rescoring.
Compare at matched realized mask and unique-fragment budgets; otherwise a coverage
gain may come only from evaluating more masks.

#### Principled replacement: canonical autoregression over cuts

Emit a cut set in strictly increasing bond-index order. At every prefix, normalize
a categorical distribution over `STOP` and the admissible bonds above the previous
index. For `S = {b_1 < ... < b_d}`:

```
P(S)
  = product_{t=1..d} P(b_t | b_1, ..., b_{t-1}, molecule, context)
    * P(STOP | b_1, ..., b_d, molecule, context)
```

Every set then has exactly one generating sequence, including the empty set through
`STOP` at the root, so no permutation marginal is needed and the distribution is
normalized. Reuse the precursor GNN encoding and condition the step scorer on the
current cut-state flags or an equivalent representation of the selected prefix.

Practical enumeration becomes beam or best-first search rather than the additive
subset-sum heap. Completed masks need not be downward-closed: the probability of
stopping after one ring cut can be low while continuing to its paired cut is high.
Fixed-width beam remains approximate and must be evaluated for search recall, but
each emitted terminal mask has an exact live model probability. Summing distinct
terminal probabilities therefore retains valid OOS accounting.

Autoregression removes the independence constraint; it does not automatically fix
the residual shortcut or the discrete-search gradient gap. Replay selected paths
with live probabilities and retain the Section 6.1.1 base-only supervision (or an
equivalent proposer objective), so candidate-specific gradients reach the
autoregressive head. Missing beam paths still receive no spectrum gradient.

#### Fourth replacement: ring-system-factored cut distribution

Correlated cleavage is confined to ring systems by graph topology: a minimal
disconnecting cut in an acyclic region is always a single bond, and generally
`|C| = 1 + (independent cycles crossed)`. That bounds where a correlated model is
needed at all, and admits a product-of-groups distribution — free bonds stay
Bernoulli, each fused ring system gets a categorical over precomputed cut patterns
— which keeps both exact normalization and *exact* top-K, unlike the mixture union
and fixed-width beam above. Both of its gates have now run: the support is affordable
only when restricted to minimal two-component cuts, and the decision gate on this
section resolves against independence (20.5% of DAG-explained intensity needs three or
more cuts, 84% of it ring-crossing). See
`masshatter_ring_system_cut_factorization.md`, which was never written (the link that stood
here pointed at a file that does not exist).

A same-ring/radius-2 pairwise Ising correction is chemically targeted, but it
requires a partition function for normalized mask probabilities and loses the cheap
exact top-K heap. Hold it until the cut-depth diagnostic and mixture probe justify
the extra inference machinery.

### 6.2 Template channel for new-bond products

Implemented as an opt-in MASShatter path with
`mass_shatter_smarts_templates: True`. SMARTS matches are cached once per unique
molecule, matching template cut sets are retained with their exact Bernoulli mask
probability, and a learned mechanism head normalizes ordinary versus template
assembly within each mask.

MASShatter owns the allowlists in `mass_shatter_templates.py`; it does not consume
`FRAG_RULES`/`NL_RULES` wholesale. The initial registry excludes broad `crf12_*`
and `crf13_2` patterns as well as pure bond-deletion and topology-redundant rules.
Shared rule objects provide the original validated SMARTS definitions. New
MASShatter-only literature rules are defined beside the allowlist so adding them
does not change the BFS preprocessing registry. Membership and the original compact
rule indices remain stable for MASShatter experiments.

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
- `NeutralLossRule` represents a new-bond product either as all precursor atoms
  except mapped neutral-loss atoms or as explicitly mapped product atoms for a
  small diagnostic ion with a variable complement;
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

#### 6.2.1 Measured scope of the channel (2026-08-19)

How much this channel must carry, measured against `src/fragnnet/frag/smarts_prepass.py`:

| | pure bond deletion | requires a new bond |
|---|---|---|
| `FRAG_RULES` (16) | 1 — `morphinan_ring_cleavage` | 15 |
| `NL_RULES` (8) | 0 | 8 |

The discriminator is a non-empty `merge_outer_map_nums`: the rule cuts into three
components and merges two outer ones. So **23 of 24 rules belong to this channel** —
xanthone, furanocoumarin/psoralen, anthraquinone, benzopyranone, phenoxazine, the
desoxymorphinans, the crf1x eliminations, and every neutral-loss rule. They are outside
the action space of *any* cut-set distribution, not merely outside a particular one, so
no change to the mask model (independent, mixture, autoregressive, or the ring-system
factorization) reaches them.

`morphinan_ring_cleavage` is the exception and needs no template: it is a pure four-bond
deletion into two components, and in 39/39 matching NIST20 molecules its cut set is
already an admissible ring-system cut pattern. Ordinary enumeration can reach it given a
cut-set model that admits correlated four-bond cuts.

**Expected coverage gain is small, and this should temper the channel's priority.**
Classifying every DAG explanation by whether its atom set is connected in the precursor,
then taking the cheapest *connected* explanation per observed peak (1,500 molecules,
23.6k spectra, 0.01 Da matching): peaks whose **only** compatible explanation is a
disconnected atom set carry **0.11%** of observed intensity. The reference DAG is
`smpTrue`, so prepass products are already in that explanation set — this is not a
measure of the prepass being absent.

Read it as: template products are almost always m/z-degenerate with some connected
fragment. The channel's value is therefore in getting the *assignment* right where a
connected explanation is coincidental, and in downstream fragmentation from correct
connectivity — not in raw m/z coverage. An adoption case resting on recall alone is
unlikely to clear a meaningful bar; the verification list below should weight
false-candidate rate and per-class intensity accuracy over aggregate recall.

Note also that `P(R | C, precursor, context)` above is the general form of an idea reached
independently from the ring-system side, where an option becomes a `(cut pattern,
assembly)` pair. The mechanism-variable formulation here is strictly more general: it
applies to free-bond cuts and to `NeutralLossRule` products whose deleted-bond set is not
a ring-system pattern at all. Prefer it. See
`masshatter_ring_system_cut_factorization.md` §9 for the measurements and §9.3 for the narrower
variant; that document was never written (the link that stood here pointed at a file that does
not exist).

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
likelihood, so it is not a foundation for the active correlated-cut work. The
[superseded RL exploration plan](masshatter_rl_exploration_plan.md) retains the
likelihood and estimator analysis; Section 6.1.2 defines the current decision path.

**Implemented 2026-08-16.** The default `bounded_additive` mode preserves all
`mass_shatter_selected_mask_limit` deterministic masks and adds up to
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
`config/massshatter/merged/ablations/bond_model/nist20_merged_mask256_gine_gsa384_bondcentric_a2_b3.yml`
and
`config/massshatter/legacy/explore/nist20_merged_mask256_gine_gsa384_bondcentric_a2_b3_explore64.yml`.

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
`config/massshatter/merged/ablations/bond_model/nist20_merged_mask256_gine_gsa384_bond2.yml`.

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

**Unique-fragment integration implemented, 2026-08-19:** compiled topology assembly
now records component-to-unique-atom-set mappings and canonical representatives.
Charge affinity is evaluated once per unique atom set, normalized per latent mask,
and marginalized across duplicate pathways before the intensity transformer. This
removes duplicate transformer tokens without discarding their probability mass or
changing OOS semantics.

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
3. **Independent bonds cannot express concerted chemistry.** Section 6.1.2 defines
   the empty-MAP downward-closure diagnostic and the matched-budget mixture probe.
   Use those measurements before committing to an autoregressive policy. A
   same-ring/radius-2 pairwise factor is chemically targeted, but normalized OOS
   accounting would also require its partition function and a replacement for the
   exact subset-sum heap.
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
   planned atom-mask/template mechanism in Section 6.2 can support their direct
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
- `docs/plans/fragnnet/dag_full_recall_tractability_plan.md` — records why static bond
  ranking inside a fixed DAG did not solve the earlier node-budget problem. The
  present proposal instead ranks complete cut masks and directly generates a
  flat fragment set.
