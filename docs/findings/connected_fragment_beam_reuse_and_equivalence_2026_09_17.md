# Connected-fragment beam: reuse and objective equivalence

> **Status:** FINDING · **Updated:** 2026-09-17 · Source and algebra review only. No new
> measurements, no training, no code changes. Speed numbers are quoted from the existing
> [beam speed audit](connected_fragment_beam_speed_2026_09_17.md) and the
> [FastGNN proposal plan](../plans/masshatter_fast_gnn_proposal.md).

## Result

Two conclusions, and they point the same way.

1. **The beam's objective is close to `cut_masks`, not a different model.** The beam score is
   the isolating mask's log probability minus the complement's intact term. Identifying the
   charge head with a fixed prior makes the two agree within a mask; the residual gaps are
   ring-reconnection terms (exactly zero for induced-acyclic fragments) and the search.
2. **Nine of roughly thirteen mechanisms in the beam already exist in MASShatter.** The one
   that dominates measured cost — complement connectivity by dense reachability — is precisely
   the one with an existing union-find replacement on both CPU and CUDA.

Because the objectives nearly coincide, the beam cannot be justified on modelling grounds; it
has to win on search cost or candidate coverage. Current measurements say it does not. That
makes the connectivity replacement load-bearing rather than cleanup.

`cut_masks` reaches the same fragments but spends its budget in mask space: the unmerged arm
enumerates up to 4096 masks for a 256 unique-fragment target. That redundancy is already logged
(`unique_components_per_mask`, `duplicate_component_pathway_fraction`), and it is partly the
price of the richer marginal rather than pure overhead, so the efficiency and equivalence
questions are one question — answerable from existing run artifacts.

The beam's own half-depth-plus-mirror design is the sharpest form of that argument. Growing
only to `ceil(atoms / 2)` and recovering the rest as complements is correct and halves the
growth rounds, but it is paid for by generating complements every round — currently by dense
reachability, which the speed audit measured as the dominant cost. Union-find yields the
fragment and its complement pieces from one pass, so the half-depth saving becomes free
instead of expensive.

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
[connected-fragment audit](../plans/masshatter_fast_gnn_proposal.md#connected-fragment-audit-2026-09-17)
records two graphs where a higher-scoring reachable fragment was lost. So the beam trades an
exact search over masks for an approximate search over fragments.

### Closing the question

Two cheap checks would settle the size of the remaining gap:

1. Histogram `p_e` restricted to ring bonds from a trained checkpoint, and evaluate
   `Σ_{e ∈ cycle bonds} p_e/(1−p_e)` per emitted fragment. The recorded
   `mask_cut_fraction_4plus = 0.77` / `cut_0 = 0.001` behaviour suggests the model is not in
   the small-`p` regime, so this should not be assumed either way.
2. Compare the beam's implicit prior `exp(−I_f̄)` against a trained charge head's actual
   component split on the same masks.

## Mechanisms that already exist in MASShatter

Reviewed [mass_shatter_fast_gnn.py](../../src/fragnnet/frag/mass_shatter_fast_gnn.py) against
the existing candidate machinery. Nine of about thirteen distinct mechanisms have an existing
counterpart:

| Beam mechanism | Existing part |
| --- | --- |
| `_connected_complements`: dense `(mols, ≤256, atoms)` reachability plus up to `max_atoms` `bmm` steps | Union-find connected components: `connected_component_atom_masks_many` and `build_component_arrays_raw` in [mass_shatter_core.pyx](../../src/fragnnet/frag/mass_shatter_core.pyx); the `connected_components` kernel via [`build_cuda_topology_device_masks`](../../src/fragnnet/frag/mass_shatter_cuda_topology.py) |
| Final boundary/internal membership assembly | `build_component_arrays_raw` / `assemble_tile` already emit `atom_indices`, `atom_component_indices`, `boundary_bond_indices`, `boundary_component_indices` in the adapter's layout |
| `_unique_rows_lex`, three call sites, `C` stable argsorts each | Hash-based atom-set dedup inside `build_component_arrays_raw`; the `deduplicate_tile` CUDA kernel; CPU `_compare_bitsets` + `_merge_sort_indices` |
| 63-bit word packing and unpacking | The codebase's 64-bit bitset convention plus `unpack_component_memberships_raw`. The 63-versus-64-bit split is gratuitous divergence |
| `_group_rank` and its keep masks | `top_mixture_bond_sets_raw` stable rank; the `rank_masks` CUDA kernel; the fragment-cutoff compaction pattern in [mass_shatter_model.py](../../src/fragnnet/model/mass_shatter_model.py) |
| Live differentiable rescoring by `index_add` of `logsigmoid` | [`_score_mask_log_probabilities`](../../src/fragnnet/frag/mass_shatter_candidates.py), which also carries mixture support |
| Input validation block | Duplicates the validation in `build_mass_shatter_candidates` and the contiguity/uniqueness checks in `_prepare_graph_batch` |
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

The construction guard in `MassShatterModel.__init__` requires unique mode, one mixture, SMARTS
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

Not guarded but unused: `MassShatterStaticGraph` (the beam rebuilds adjacency every step) and
`candidate_backend` / `mask_selection_threads`, so the CUDA selection and topology backends are
unreachable from this proposal method.

Separately, the beam's timing record fills six of ten fields with `0.0` and sets
`output_transfer_ms` equal to `probability_scoring_ms`, so beam-versus-`cut_masks` timing rows
are not like-for-like.

## Budget efficiency: `cut_masks` does the same work, in mask space

`cut_masks` reaches the same fragments, but spends its budget on **masks** rather than
fragments, and a fragment `f` is produced by every mask whose cut set contains `∂f` and leaves
`f` connected — exponentially many. The repository already prices this in: the unmerged arm
sets `mass_shatter_ranked_mask_budget: 4096` against `mass_shatter_unique_fragment_target: 256`,
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

One further source-level observation, not profiled here: the reachability loop breaks on
`th.equal(updated, reach)`, which returns a Python bool and therefore forces a device-to-host
synchronization on **every iteration**, inside every chunk, inside every round. `_search` also
synchronizes at `recorded.any()`, at `int(counts.max().item())`, at `int(keep_rank.max().item())`
and at each boolean-mask index. On a 76-atom batch at half depth this is plausibly hundreds of
syncs per batch. This is the failure mode the
[MASShatter GPU audit](../plans/masshatter_gpu_audit.md) catalogues, it is independent of the
union-find question, and it is cheap to test.

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
checks with the union-find components MASShatter already has is what decides whether the beam
is a cheaper route to nearly the same objective or a more expensive one.

## Limits

- Source and algebra review at `08e128d5`.
  `src/fragnnet/frag/mass_shatter_fast_gnn.py` is clean at that commit;
  `mass_shatter_candidates.py` and `mass_shatter_model.py` carry unrelated uncommitted changes.
- No benchmark, training run, quality evaluation or test execution was performed for this
  document, and no code was changed. Every performance number is quoted from the linked
  documents and inherits their contracts and limits.
- The ring-gap magnitudes are worked examples at assumed `p_e`, not measurements on a trained
  checkpoint. The two checks above are unrun.
