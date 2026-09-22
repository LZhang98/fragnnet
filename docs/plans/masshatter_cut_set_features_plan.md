# MASShatter candidate features: what pooling destroys, and the shortcut it leaves behind

> **Status:** PARTLY IMPLEMENTED — B0, an adjacency form of A, and G are in the tree behind
> default-off keys, five matched arms built, nothing trained &nbsp;·&nbsp;
> **Updated:** 2026-09-22
>
> **The premise numbers below are superseded.** They came from the depth-3 BFS DAG; the
> [candidate-descriptor measurement](../findings/masshatter_candidate_cut_descriptors_2026_09_22.md)
> recomputes them on MASShatter's own candidates (86.9% multi-cut, 73.6% same-ring, 68.4%
> adjacent) and redirects item A from ring topology to adjacency.
>
> Candidate-level changes for MASShatter's intensity head, each aimed at a specific shortcut,
> plus a sweep of everything else the pipeline computes and throws away. Unlike the descriptor families closed in
> [feature headroom](../findings/feature_headroom_2026_09_22.md), these are **not** derivable by
> the encoder, because the information is destroyed *inside the head* by mean pooling or is
> discarded before it gets there. **96.8% of fragments are cut at ≥2 bonds, and 55.4% have two
> cuts sharing a ring** — a ring opening — which the current token represents identically to two
> unrelated cuts.

## Why this is not the closed question

[Feature headroom](../findings/feature_headroom_2026_09_22.md) showed that molecular descriptors
are recoverable from what the encoder already receives, so adding them buys nothing. That result
is about *molecular* features reaching the *encoder*. It says nothing about the two failure modes
below, which are different in kind:

- information the model **computes and then discards** before the intensity head, and
- information destroyed by **mean pooling inside the token**, which no upstream capacity restores
  because pooling is applied after the fact.

Both matter for MASShatter specifically, because the
[2026-09-22 read-out](../findings/masshatter_overfit_readout_2026_09_22.md) puts ~98% of the
generalization gap in *intensity allocation among peaks the model already proposes* — which is
exactly what the candidate token drives.

## What the candidate token knows about the cut today

From `CandidateIntensityTransformerHead.forward`: three pooled embeddings (fragment, complement,
boundary), context, `cut_counts`, `log1p(component_sizes)`, base-probability features, and since
2026-09-20 the mass and Gasteiger-charge scalars.

Of these, everything describing **what was broken** is either a scalar count (`cut_counts`) or a
**mean** over per-cut-bond encodings. Mean pooling is permutation-invariant and additive, so any
relationship *between* the cut bonds is gone by construction.

## Measurement: the destroyed distinction is the common case

`scripts/debug/debug_cut_set_topology.py`, 200 molecules, 85,819 depth-3 DAG fragments with
boundary bonds:

| | share |
|---|---:|
| fragments cut at exactly 1 bond | 0.032 |
| fragments cut at ≥2 bonds | **0.968** |
| of the multi-cut ones, ≥2 cuts **share a ring** (ring opening) | **0.572** |
| of the multi-cut ones, ≥2 cuts **share an atom** (adjacent) | 0.350 |
| as a share of all fragments: same-ring / adjacent | 0.554 / 0.339 |

**Caveat, stated up front:** these fragments come from the depth-3 BFS DAG, where cut count is
bounded by enumeration depth, so the *distribution* (77% at exactly 3 cuts) is an artifact of that
enumeration, not MASShatter's. MASShatter ranks unrestricted multi-bond masks and sees at least as
many multi-cut candidates. The distribution must be recomputed on real candidate batches before
being quoted; the qualitative point — multi-cut candidates dominate, and ring-sharing among cuts
is common — is what carries.

## The shortcut this leaves

With only a count and a bag of pooled bond encodings, the only cut descriptor with usable variance
is **how many bonds were broken**. The head is therefore pushed to express the cost of
fragmentation as a function of cut count and fragment size — a size/count law that fits the
training distribution and encodes no chemistry. Two cuts in one ring (a ring opening, which
releases nothing on its own, and the retro-Diels-Alder signature) and two cuts in unrelated
positions are the same object to it.

## Proposal

### A. Cut-set topology scalars — REDIRECTED TO ADJACENCY, 2026-09-22

Originally proposed as three scalars from the boundary bond indices plus the precursor's ring
info: `n_ring_shared_cut_pairs`, `n_adjacent_cut_pairs`, `n_distinct_rings_cut`.

**What the measurement says.** On MASShatter's own candidates, given the summed cut-bond vector
that B0 now feeds, the residual ambiguity is **0.000 for the ring-sharing flag** and **0.489 for
the adjacency flag**. The sum carries `b_ring` counts, so "two of the cuts were ring bonds" is
already in the token, and on this population it never disagrees with "two cuts share a ring".
Bucket resolution: cut-bond sum 22.55 candidates per within-molecule bucket, adding adjacency
10.44, adding the two ring terms on top **10.41**.

**And the ring terms are not free.** SSSR ring identity per bond is *not* on the graph — only the
`b_ring` flag and, in the ring-topology arm, `b_ring_size`. Shipping them means a new graph field
and a molecular-cache regeneration, for 0.02–0.03 of bucket resolution.

So A ships as **`mass_shatter_candidate_cut_adjacency_features`**: two columns, the number of
cut-bond pairs meeting at a shared atom, split by whether that atom is retained or lost. Two
bonds share at most one atom, so summing `C(count, 2)` over the (component, atom) incidence counts
each pair exactly once. The split is worth its column (12.28 -> 10.44) and separates two different
events: a shared retained atom means one atom of the fragment lost two substituents, a shared lost
atom means the fragment expelled a bridging atom — the CO / CH2 / S extrusion shape.

Adjacency is also the cleanest case of this plan's thesis: it is a relation *between* cuts, so a
mean over per-bond encodings cannot represent it even in principle — a mean cannot detect that the
same atom appears twice.

### B0. Pool the cut bonds' own features into the candidate token — the safe version

`_boundary_embeddings` encodes each severed bond as
`boundary_bond_encoder(atom_embeddings[inside], atom_embeddings[outside])`. It has
`boundary_bond_indices` in hand and **never reads `mol_pyg.chemical_bond_attr`**. So the cut
bond's own features — type, `b_ring`, `b_conj`, and the `b_ring_size` / `b_fused_ring` added on
2026-09-22 — reach the intensity head only through endpoint atom mixing, and an internal atom
mixes *all* of its incident bonds into one state. For any endpoint of degree > 1 the specific cut
bond's attributes are not cleanly recoverable.

Pool `chemical_bond_attr[boundary_bond_indices]` per candidate, **sum and mean**. The sum is the
mechanically meaningful one: it is the *multiset of broken bond types* — "two aromatic ring bonds
and one single bond" — and it delivers "how many cut bonds were ring bonds" for free, which is
part of what A is reaching for.

Cost is 9 columns at the `c4a_base_s1` bond width, 16 with the ring-topology arm, doubled for
sum+mean, on a ~1,285-dim token. **No coupling to the proposal at all**, because these are raw
input features rather than learned states. This subsumes most of the concrete information gap in
B at none of its risk, and it unblocks bond features that are already in the model but cannot
reach the head that allocates intensity.

### B. Feed the break head's learned bond states to the intensity head

`BondFocusedBreakHead` runs `mass_shatter_bond_num_layers: 3` of alternating bond↔atom updates and
maintains **explicit per-bond hidden states**. It is the largest module in the model (3.868M of
9.22M parameters at the `c4a_base_s1` geometry). It returns **only `bond_logits`**; the states are
discarded. The intensity head's boundary embedding is instead built from *atom endpoint*
embeddings (`chemical_bond_embeddings` is literally the mean of the two endpoint atom embeddings,
`mass_shatter_model.py:2497`).

So the richest representation of "what bond is this" in the model never reaches the head that
decides how much intensity the cleavage gets. The [feature-set audit](../reference/masshatter_feature_set_audit.md)
§7.4 listed exactly this as an option — "Pooling those existing bond states as the boundary source,
or porting `CutBondEncoder`" — and the atom-based `CutBondEncoder` was the one implemented.

**What the learned states add over B0.** The bond state is initialized from
`2 * atom_dim + bond_feature_dim + context_dim` and refined by `num_layers` rounds of alternating
bond↔atom propagation, so bonds sharing an atom exchange information — a line-graph pass the
atom-level GNN does not perform. More importantly it is **context-conditioned**, and
`mass_shatter_context_in_gnn` is `False`, so the precursor GNN's atom embeddings are
**CE-invariant**. The break head's bond states are therefore the *only* CE-conditioned bond-level
representation anywhere in the model — and the repository's own decomposition puts the gap in
CE-dependent allocation. That is a stronger argument than this plan originally gave B.

**But B is in tension with C.** Those states are the proposal's pre-logit representation, so
handing them to the intensity head lets it read proposal-correlated information — a richer form of
exactly the `mask_log_probabilities` shortcut that C removes. Detaching stops the gradient, not
the reading. Two ways out, in order of preference:

1. give the intensity head its **own** small context-conditioned bond encoder over the cut bonds,
   so the CE-conditioned bond representation exists without sharing the proposal's; capacity is
   [not the bottleneck](../findings/feature_headroom_2026_09_22.md), so the duplicated parameters
   are an acceptable price for a clean separation;
2. share the states but **detached**, with precedent in `mass_shatter_detach_base_ion_h` and
   `frag_competition_detach_encoder` — cheaper, but it does not prevent the head from reading the
   proposal's ranking out of them.

### C. Remove a shortcut rather than add a feature — ALREADY OFF IN THIS BASE

`use_base_probability_feature` puts `mask_log_probabilities` *into* the candidate token, so the
intensity head can read the proposal's confidence directly instead of judging the candidate. In
unique-fragment mode its companion `charge_log_probabilities` is already identically zero (audit
§7.5). Turning it off is a **config-only** experiment and the cheapest of the three.

**CORRECTION, 2026-09-22.** There is no key to turn off: `mass_shatter_pl.py:1008` sets
`use_base_probability_feature=(not mass_shatter_proposal_anchored)`, and `c4a_base_s1` — the base
this plan diffs everything against — runs `mass_shatter_proposal_anchored: true`. So the candidate
token in the whole `c4a` family already excludes `mask_log_probabilities`, and its width is 1,285
= 3x384 + 128 + 2 + 2 + 1 (verified by constructing the head), with no base-probability column.
C is therefore **not available as an arm here** and is not the cheapest first step; it would be a
question for a `proposal_anchored: false` lineage instead. The tension this created with B is also
weaker than stated, though B still shares the proposal's pre-logit states.

### G. Per-atom dropout on the pooled component embeddings

The other half of the same idea: having given the token a description of the *cut*, reduce what it
leans on from the *whole component*. The first two blocks of the token are mean pools over the
candidate's atoms and over its complement — 768 of 1,285 columns — and they are the most
identifiable thing the head sees. On 882 molecules / 57,925 fragments the cc atom mask resolves
**1.00 distinct fragments per descriptor bucket** (100% singletons) against 19.67 for a cut-bond
multiset, which is why the same probe found the *least* identifiable arm generalized best.

`mass_shatter_candidate_atom_dropout` drops each atom from a candidate's membership set with
probability p before pooling, in training only. A candidate that loses every atom keeps all of
them, since an empty pool has no mean.

**The complement is deliberately not held exact.** It is derived as precursor minus candidate, so
an exact complement beside a corrupted candidate leaves the true candidate sum recoverable by
subtraction — exactly the memorization route this targets. Dropped atoms therefore migrate to the
complement side, and both pools stay means over a well-formed atom set. This is where it departs
from FraGNNet's `cc_atom_dropout`, which has no derived complement to leak through. Sizes, masses,
charges, compositions, cut counts and the boundary embedding all keep the **true** membership, so
the intervention is confined to the pooled component embeddings.

It costs no parameters and no token width, and it is inert at evaluation, so validation metrics
stay deterministic. FraGNNet's written p = {0.0, 0.1, 0.2} sweep of the same hypothesis has never
been run.

## Also unused, found 2026-09-22 by sweeping the pipeline for discarded computation

### D. Free-valence capacity is a hard mask and nothing else

`atom_free_valence_slots` is attached to every molecular graph and read in exactly one place:
`_transferable_hydrogen_shifts`, where it is summed per candidate to bound how many hydrogens the
fragment can absorb (`valence_gain = fragment_slots + fragment_cut_counts`). It never reaches the
intensity head.

Note the asymmetry: the *other* term in that same bound, `fragment_cut_counts`, **is** a token
feature. So the head knows how many bonds were cut but not how many hydrogens the fragment could
take — while the hydrogen-shift ladder is an output dimension it has to allocate probability over.
One extra scalar, already summed, no coupling.

### E. Merge multiplicity — DEPRIORITIZED

Merged targets pool 1 to 112 spectra (mean 6.1, p50 3, p90 15), 43.6% of them from a single
spectrum, and every target enters the loss with equal weight while `_embed_ce_values` mean-pools
the CE list so the count is invisible. All true, and **not worth acting on**: merged mode is a
development harness for faster iteration, not the production target. Recorded so the measurement
is not repeated.

### F. Known and prepared, not run: the gated attention mechanisms

Production sets `mass_shatter_intensity_transformer_attention: induced`, and the head's own
validation refuses four options under it — `mass_pair_bias`, `ce_pair_bias`, `ce_film` and
`qk_norm` all require `full`. So the model currently cannot use a **neutral-loss-aware relational
prior over candidate pairs** (`_NEUTRAL_LOSS_CENTRES` covers H2, NH3, H2O, CO, C2H4, CO2, HCOOH)
or **CE-conditioned attention** — the latter aimed squarely at where the decompositions put the
gap. Arms `a2`-`a7` in `config/massshatter/ma_mi_merged/attention_2026_09_21/` already exist;
[their record](masshatter_attention_budget_2026_09_21.md) says BUILT, NOT RUN.

### Checked and *not* findings

Recorded so they are not re-chased.

- **`charge_log_probabilities` zeroed in the candidate token** is correct design, not a loss. On
  the unique-fragment path the charge term is added into the joint log-probability instead
  (`mass_shatter_model.py:2573`), so the information is applied rather than discarded.
- **`mass_shatter_intensity_atom_decoder_num_layers`** is not a vestigial key. `mass_shatter_pl.py:936`
  raises if it is non-zero — a deliberately retired path with a guard, not an unexplored option.
- **The CE ladder** survives better than a scalar mean would: `_embed_ce_values` embeds each
  reading *before* averaging, so the shape of the set is partly retained. Only the multiplicity is
  lost, which is E.

## Which of these merged screening can and cannot answer

Merged mode is a development harness. Every arm built in the 2026-09-21/22 campaigns is
merged-only — `config/massshatter/ma_mi_unmerged/` holds just `baseline`, `cmf`,
`context_ablation` and `unique_fragment` — and so is the evidence this plan leans on: the
train `wrecall` 0.8975 / validation 0.8911 against a 0.151 cosine gap comes from the merged
read-out. The [unmerged audit](../findings/masshatter_unmerged_overfitting_2026_09_16.md) shows a
growing generalization gap on its 27 runs too, so the direction agrees, but there is **no unmerged
coverage-versus-allocation decomposition**; do not assume the 98% figure transfers.

The split that matters for screening:

- **CE-neutral — merged screening is a fair proxy.** B0, A, C and D are about representing the cut
  and the fragment. Nothing about them depends on how the target was assembled.
- **CE-sensitive — merged screening may mislead.** F (`ce_film`, `ce_pair_bias`) and the CE half of
  B's argument. A merged target is CE-consensused and the model sees a bag-mean CE, so the work a
  CE-conditioned mechanism has to do is *smaller* there than on unmerged rows, where every row
  demands one specific energy's allocation. A merged null for these is weak evidence; they deserve
  an unmerged arm before being written off.

## Order of work

0. **F** costs nothing to decide — the arms exist. Whether they run is a scheduling call, not an
   implementation one.
1. ~~**C**~~ — withdrawn as an arm; `c4a_base_s1` already runs with the base-probability column
   absent (see the correction above).
2. **B0** (built, `e1_cut_bond_feats`) — the cheapest real information gain, no coupling, and it
   unblocks bond features the model already computes.
3. **G** (built, `e2`/`e3`) — independent of B0 and the other half of the same idea, so the two
   can screen in parallel. Run both rates: one point cannot separate a null mechanism from a rate
   that is too weak.
4. **A**, as adjacency (built, `e4`; stacked with B0 as `e5`). `e4` - base is what adjacency is
   worth alone, `e5` - `e1` is what it is worth given the bond sum, and the 0.489 residual
   ambiguity is a statement about the second. If `e5` lands at `e1` or at `e4` rather than above
   both, the cut description saturates and A closes with it.
5. **D** is independent of the cut-set line and can go in any screen — one summed scalar. **E** is
   dropped.
6. **B** last and only if B0 reads positive, preferring a separate bond encoder over sharing the
   proposal's states, since it is the only option that changes what flows between heads.

Each as a one-key diff from `c4a_base_s1` at the same seed, matching the `c4a` campaign
convention, screened at **128 epochs** — confirmed in the
[2026-09-22 read-out](../findings/masshatter_overfit_readout_2026_09_22.md), which measures a
median +0.0031 between epoch 128 and best across 15 arms, at the replicate floor.

## Implementation state, 2026-09-22

Both landed keys are **off by default**, so no existing comparison moves under them.

| key | arm | what it does | width |
|---|---|---|---|
| `mass_shatter_candidate_cut_bond_features` | `e1_cut_bond_feats` | B0: pooled `chemical_bond_attr[boundary_bond_indices]`, sum and mean | +2x bond width (18 here; 1,285 -> 1,303) |
| `mass_shatter_candidate_cut_adjacency_features` | `e4_cut_adjacency`, `e5_cut_bond_plus_adjacency` | A: cut-bond pairs meeting at a shared atom, split retained/lost | +2 |
| `mass_shatter_candidate_atom_dropout` | `e2_cc_atom_drop01`, `e3_cc_atom_drop02` | G: per-atom dropout on the component pool, training only | none |

Code: `MassShatterModel._cut_bond_features`, `._cut_adjacency_features` and
`._candidate_atom_dropout`, wired into **both** candidate paths (unique-fragment and legacy
pathway-token); `CandidateIntensityTransformerHead` gains `cut_bond_feature_dim`,
`cut_adjacency_feature_dim` and their two arguments, validated the same way as the composition
block. The adjacency helper reuses `boundary_bond_endpoint_orientation`, which repeats the
membership sort `_boundary_embeddings` already does when the boundary encoder is on — one sort of
the membership relation, an item for the batching audit rather than a correctness concern.
Configs:
`config/massshatter/ma_mi_merged/candidate_token_2026_09_22/`, each verified to be a **one-key
diff** from `c4a_base_s1` after template resolution, same seed, same 128-epoch box.

Validation: 19 new tests in `tests/test_mass_shatter_candidate_features.py` (54 pass in that
file), full suite green, Ruff and Black clean on every changed file. **Nothing has been trained.**
Both feature keys change the token width, so a checkpoint trained without one will not load into
an arm that has it.

**Smoke coverage.** `config/debug/debug_tiny_mass_shatter_cut_features.yml` stacks all three keys
at toy width with `candidate_atom_dropout: 0.3`, so one run exercises `_cut_bond_features`,
`_cut_adjacency_features` with its endpoint-orientation lookup, `_candidate_atom_dropout`
including the restore-an-emptied-candidate branch, and both widened token blocks. Two epochs on
`debug_tinyv4`, 64 spectra, `detect_anomaly: true`, CUDA candidate backend, unique-fragment mode:
green, no NaN, loss falling (train -0.509 -> -0.754, validation -0.806 -> -0.981).

Decomposed, one key at a time against the all-off control at three seeds, because the stacked run
first looked slightly behind its control and that reading did not survive the noise floor:

| arm | train loss | val loss | val cos |
|---|---:|---:|---:|
| all off, seeds 42 / 43 / 44 | -0.7705 / -0.7248 / -0.7363 | -1.0015 / -0.9585 / -0.9346 | 0.2173 / 0.1960 / 0.2248 |
| cut-bond features only | **-0.7921** | -0.9998 | 0.2163 |
| adjacency only | -0.7583 | **-1.0038** | 0.2228 |
| atom dropout 0.3 only | -0.7246 | -0.9814 | 0.2050 |
| all three | -0.7544 | -0.9808 | 0.2078 |

Every arm sits **inside** the control's own seed band, and the stacked-vs-control gap (0.0095 val
cosine) is a third of the seed spread (0.029). The feature blocks do not slow the fit -- cut-bond
features give the best train loss of the seven runs and adjacency the best validation loss --
which is expected, since changing the token width also reshuffles the initialization RNG stream.
The only systematic effect is the dropout suppressing the training fit, which is a regularizer
doing its job. None of this is a measurement of the features; 64 spectra and two epochs cannot be.

The blocks are live rather than inert, and their cost is confined to the token:

| arm | full model, test width | delta |
|---|---:|---:|
| neither | 14,450 | |
| cut-bond features | 14,738 | +288 = 2 x 9 x 16 |
| cut adjacency | 14,482 | +32 = 2 x 16 |
| both | 14,770 | +320, exactly additive |
| atom dropout only | 14,450 | +0, a regularizer adds no parameters |

At the `c4a_base_s1` geometry the token goes 1,285 -> 1,303 (cut-bond), 1,287 (adjacency), 1,305
(both).

## What would falsify the premise

Partly answered without a checkpoint: the
[candidate-descriptor measurement](../findings/masshatter_candidate_cut_descriptors_2026_09_22.md)
shows the information is genuinely absent from the token rather than already encoded — but note
its own warning, that the 2026-07-23 probe found the *least* identifiable fragment descriptor
generalized best. Resolving more candidates is evidence the head cannot otherwise obtain the
information, not evidence that giving it raises validation cosine.

The model-side version still stands. If the candidate token is already achieving whatever these
encode, a corruption probe would show it: randomize `cut_counts` and the boundary embedding on a
trained checkpoint and measure the prediction shift. Large shift means the head leans on the cut description it has; no shift means
it has already stopped listening, and none of A/B/C will matter. That probe needs a production
checkpoint, which is not in the local tree. **Nothing here has been implemented or trained.**
