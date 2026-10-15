# MASSter: rearrangement rules with continuation in training

> **Status:** PROPOSED · **Updated:** 2026-10-15
> Nothing built. Phase 0 is model-free or CPU-only gates (reach ceiling with a null-matched control, and a
> checkpoint-side after-selection expansion with a direct parent-survival measurement). Phase 1 is one
> backend-agnostic training stage that applies admitted rewire rules to every kept fragment, `mask_order`
> selection only. "After selection" first is a COST-ORDERING decision, not a ruling on the user's
> "after selection or as part of selection": in-selection stays open (Phase 2). No rule is admitted yet:
> the "Admitted rules" table is the OUTPUT of gate G0 and Phase 1 is blocked on it and on the three
> decisions in "Decisions that gate Phase 1" (each has a default).

Related: [SMARTS pathway plan](masster_smarts_pathway_plan.md) (closed null; templates have no
continuation), [CUDA audit](../../findings/masster/masster_smarts_cuda_audit_2026_09_27.md),
[fragment marginal selection plan](masster_fragment_marginal_selection_plan.md), hub
[MASSter generalization](../../workstreams/masster_generalization.md).

## Objective and evidence

Make mechanism rules with continuation part of MASSter training. A rule is a mapped SMARTS plus a graph
edit (delete atoms, cut bonds, form bonds), as in
[`debug_masster_rewire_rule_gate.py`](../../../scripts/debug/smarts/debug_masster_rewire_rule_gate.py)
(`_edit_edges`, `_matches`, `_reaches`). Its products are the connected sets of the rewired graph
G' = (G - deleted - cut) + formed, i.e. the rule applied to any connected fragment that contains the matched
motif, followed by ordinary cuts. The user's instruction: the rule may be applied to all fragments after
fragment selection, or as part of selection.

Today a MASSter template (`candidates/smarts_templates.py`, `masster_smarts_templates: False`) forces the
template's cut mask into the pool and appends each DISCONNECTED side as one whole fragment. It has no
continuation, and its rule identity feeds only the append-only `MASSTER_RULE_VOCABULARY` (hashed in
checkpoints). The `t1i`/`p1` arms were stopped at G1 (+0.196% of bearing-spectra intensity), so this plan
is gated on a model-free ceiling and a null control before any GPU time.

Mining (2026-10-14, [`output/rearrangement_mining_2026_10_14/`](../../../output/rearrangement_mining_2026_10_14),
script `scripts/debug/smarts/debug_masster_rearrangement_mining.py`): Orbitrap [M+H]+ spectra of the
MassSpecGym benchmark and NIST20 (`nist20_inchikey`), train+val only, 29,802 molecules, 282k spectra. A
peak >= 5% of base is a "rearrangement" peak when no connected atom set of the precursor has its heavy
composition (own-H within +-4), so no ordinary bond-cut fragment can make it.

| Quantity | Value |
| --- | --- |
| Share of all spectrum intensity on rearrangement peaks | 1.1-1.4% |
| Spectra with at least one rearrangement peak | 10% |
| Spectra with >= 25% of intensity on them (the tail) | 1.1-1.9%; 939 molecules, 3,711 spectra |
| Tail spectra with a rearrangement base peak | 57% |
| Tail intensity in ions keeping 30-70% of heavy atoms | 72% |
| Loss class (>= 25% spectra) | S+O 43.5%, N+O 18.5%, halogen 13.2%, O-only 7.6%, P 6.7%, S-no-O 4.7%, N-only 4.3%, C/H-only 1.6% |
| P(tail \| motif) | sulfonyl 14.7%, P 15.1%, nitro 8.1%, permanent charge 6.9%, halogen 4.4%, base 3.2% |
| Tail share vs NCE | rises with NCE (median per-molecule corr 0.60) |

Canonical location: these numbers belong in a finding (proposed
`docs/findings/masster/masster_rearrangement_mining_2026_10_14.md`, to be written by the main session
with the cost numbers from G0/G0b; this plan then links to it and drops the restated table). Until it
exists this table is the only record. Source files: `output/rearrangement_mining_2026_10_14/`
(`rearrangement.pkl`, `peaks.pkl`, `molecules.csv`).

Validation size (the evaluation is validation only; the counts above are train+val). Recomputed from
`rearrangement.pkl` and `molecules.csv` on 2026-10-14: validation has 60,228 spectra of 6,464 molecules
(NIST20 4,222 molecules / 52,020 spectra, MassSpecGym 2,242 / 8,208). The validation TAIL (>= 25% of
intensity on rearrangement peaks) is **669 spectra from 194 molecules** (NIST20 582 spectra / 150
molecules, MassSpecGym 87 / 44), 1.11% of validation spectra; the secondary tail (>= 10%) is 1,836 spectra
/ 474 molecules. Mean rearrangement share inside the validation tail is 0.405 (median 0.356); the whole
validation split carries 1.02% of its intensity on rearrangement peaks, 0.45% of it from tail spectra.
Paired bootstrap resamples the 194 tail molecules, not the 669 spectra.

Consequences. (1) The 72% figure means products need a rearrangement FOLLOWED by ordinary cuts, so
whole-precursor templates are insufficient. (2) The tail is 1.1% of validation spectra. Arithmetic, not a
measurement: whole-split cosine moves by about (tail fraction) x (tail delta) = 0.011 x delta, so even a
+0.05 tail delta gives +0.0006, far below the 0.0042 seed spread. The tail metric decides, with whole split
as a no-regression guard. The 0.0042 is a whole-split spread; the TAIL's seed spread is unmeasured
(Phase 0, step P-1) and thresholds below are tied to it.

## Admitted rules

This table is the output of G0, so it is empty by construction and is NOT a plan defect to be filled by
hand: no rule is admitted before G0 runs, Phase 1 cannot start with it empty, and the plan is complete as
PROPOSED because the admission procedure is fixed. A rule is admitted only if its real-match reach exceeds
its `null_smarts`-matched reach (bootstrap CI over molecules excludes 0) and it covers >= 1% of tail-spectra
intensity. A bare halogen rule matches 31% of molecules and would flood the cap; it needs a tight SMARTS.
Candidate families to encode and test, from the mined loss classes (not admitted): S+O (sulfonyl SO2
extrusion, S-to-O migration), N+O (nitro), P, halogen (tight SMARTS only), O-only.

| Rule | Loss class | Delete / cut / form | Real reach | Null reach | CI of diff | Matches per matching mol | Admitted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (empty until G0 runs) | | | | | | | |

**Provisional input to G0 (2026-10-15).** The unlimited-pool reach (R_all, the gate's rewired-graph
search) of 13 rules that passed per-class review is measured in the
[rearrangement mining finding](../../findings/masster/masster_rearrangement_mining_2026_10_14.md#mechanism-rules-that-pass-final-combined-run-2026-10-15).
Together they reach 39.0% of rearrangement intensity (null 6.8%), and 41.1% of the validation tail's
(null 6.7%). The tail cosine ceiling moves from 0.652 to 0.782. R_all is above the 10% pass bar, but the
bar is on R_after at depth 1, which G0b measures. The rule list for G0 is
`output/rewire_rule_gate_2026_10_15/survivors.json`. The table above stays empty until G0 runs.

Measured match rates (3,000 random molecules, RDKit uniquified): sulfonyl 7.1% of molecules (1.05 matches
among matching, max 2), nitro 2.7% (1.12), P(IV) 1.7% (1.29, max 6), halogen 31.3% (1.73, max 8).

## Design

### Rule representation

New module `src/fragnnet/models/masster/candidates/rewire_rules.py`, kept separate from
`smarts_templates.py`.

- `REWIRE_RULES`: `{name, smarts (mapped), delete, cut, form, allowed_precursor_types}`, ported from the gate
  schema and restricted to admitted rules. `delete` removes every bond of the listed atoms (they leave in the
  neutral).
- `REWIRE_RULE_VOCABULARY` (append-only), `MASSTER_NUM_REWIRE_RULES`, `REWIRE_RULE_NAME_HASHES`. Rules are
  compiled in, so a checkpoint cannot carry the rule list, only a fingerprint of it. The existing
  `MASSTER_RULE_NAME_HASHES` scheme hashes the NAME only; here the hash is CRC-32 of the canonical edit spec
  `name | mapped SMARTS | sorted delete | sorted cut | sorted form | allowed_precursor_types`, so changing a
  rule's SMARTS or edit while keeping its name changes the hash and the load pre-hook raises (strict-prefix
  check on spec hashes). The name is kept separately in the exported provenance column. The golden-instance
  test is then a CI aid, not the only protection: a drifted rule fails on checkpoint load. The vocabulary is SEPARATE from `MASSTER_RULE_VOCABULARY`. Appending to
  the latter changes `MASSTER_NUM_RULES`, hence the shape and `fork_rng` init of `rule_embedding` and
  `rule_pathway_residual` for every newly built SMARTS model, which breaks feature-off bit-identity for
  SMARTS-on models (old checkpoints would still load by the prefix rule).
- Frozen `RewireMatch(rule_index, motif_atoms, deleted, cut_pairs, form_pairs)` in molecule-local atom ids.
  `build_rewire_matches(mol, max_matches_per_rule)` runs on the standardized molecule (the same `mol_smiles`
  `_cached_masster_templates` parses), uniquifies on the edit atom tuple, sorts deterministically and
  truncates with a returned truncation count. `filter_rewire_matches` mirrors `filter_masster_templates`
  (precursor-type guards; MS/MS mining covers [M+H]+ Orbitrap only).
- Cut pairs are resolved to bond ids from `chemical_bond_index` at stage time. A pair with no bond id is a
  `ValueError` (rule/graph mismatch), never a skip.

### Where rules apply: phasing

Let F be a component of G - m (an ordinary fragment). The rule applied to F gives the components of
F' - (m' within F). Applying rules to every KEPT fragment is therefore a subset of applying them to every
connected set of G (the G0 unlimited-pool ceiling), and it differs from in-selection (ranking masks over the
rewired graph G' with the rule's cuts forced) in two ways: (a) it can only act on parents F that the
ordinary ranking kept, and (b) it cannot reach products whose ordinary cuts remove motif context before the
rule. The precursor is always a kept fragment (empty mask, cut count 0), so precursor-level rules always
fire; asserted by a test rather than assumed.

What is kept under `masster_fragment_selection: mask_order` with `masster_unique_fragment_target: 256`
(i1): the builder enumerates cut masks in exact Bernoulli order (budget `masster_ranked_mask_budget: 4096`)
until 256 UNIQUE atom sets exist; the keep mask is then the relative-probability floor (0.0 in i1, so all
kept) plus the valid-fragment fallback. So F survives iff its cheapest mask ranks within the first 256
unique sets. The ranking CAN drop F: a large F needing several cuts ranks below many one-cut sets, and a
motif inside a 30-70% fragment can require F to be a 2-3-cut set. Hence the Phase 2 trigger is measured
directly as an F SURVIVAL rate (below), not inferred from the intensity ratio G0b/G0, which conflates
survival with continuation reach.

| Phase | Variant | Why |
| --- | --- | --- |
| 0 | Model-free reach ceiling (after-selection on an unlimited pool and in-selection ceiling) and checkpoint-side expansion with F survival | Decides whether any GPU time is justified |
| 1 | After selection, one torch stage in the model, no builder or kernel change | Cheapest first training read (a cost ordering, NOT a design verdict); bit-identical off |
| 2 | In selection: conditional top-k S over the rewired graph with T forced | If Phase 0 shows F survival is the limit; the user's other option, still open |

v1 supports `masster_fragment_selection: mask_order` ONLY, with a `ValueError` guard rejecting `marginal`.
Reason: marginal requires `proposal_selection_only`, SMARTS off and a closed-form Bernoulli release score
(`masster_model.py` ~1625-1640), its pool-coverage outputs and release scores are defined on ordinary
fragments only, and a rewired set has no release score. The stage is mechanically placeable after the keep
under marginal, but combining them is a separate design (it changes the coverage loss), so it is out of v1.
The control is the existing `i1` lineage (`i50ig7rf`, seed 397623).

Rejected for v1: the "forced `T u {b}` masks" variant (rewire templates through the existing forced-mask and
disconnected-side path). Its continuation is one data-independent cut, it inflates NODE count by forced masks
unrelated to model scores, and it requires the stopped t1i machinery plus a vocabulary append. Its
experimental scaffolding (G0 gate, null control, kill rules) is adopted here.

### Required model configuration (guards, all `ValueError`)

Rewire v1 requires: `masster_unique_fragment_mode`, `masster_fragment_selection: mask_order`,
`masster_proposal_selection_only: true` (i1 sets it, line 235 of `i1_inst_ob_obf.yml`), `masster_h_transfer_bound: valence`,
candidate backend `cpu` or `cuda` full. It is rejected with: `marginal`, `connected_fragment_beam` or the
hybrid backend, `fragment_connectivity_encoder`, `intensity_candidate_graph`, `use_base_probability_feature`
(already excluded by `proposal_selection_only`). On the check that every existing config enabling rewire
carries `h_transfer_bound: valence`: the template default is `valence` (`template.yml:812`) and `i1` sets it
explicitly, but no config enables the new flag today, so the guard is the real protection and the arm
configs are derived from `i1`. Old checkpoints without the new keys get the template's OFF values.

Why `proposal_selection_only` is required: the intensity head's logits are then the residuals alone
(`corrected_fragment_logits = fragment_residuals`, `masster_model.py` ~3917-3920) and the proposal
log-probabilities never enter the intensity softmax. That makes the extras' "base log-probability" a
non-quantity in a0, which removes the prior-choice problem (see the stage, step 9) and keeps the proposal
outputs (computed from ordinary fragments before the append) untouched. Under a non-selection-only model
the intensity head receives `fragment_log_probabilities` as a prior and a base value for extras would have to
be chosen (0, the parent's, or a learned constant), changing every matching molecule's prediction; v1
refuses that case rather than choosing.

Interaction with other features that read the candidate set (decided, each with a test):

- `intensity_candidate_graph` (`candidate_graph_edges`, edges from shared atoms in membership):
  FORBIDDEN in v1. Extras have non-nested atom sets relative to the ordinary parents (they lack the
  deleted atoms), so parent/child/complement edges are undefined for them. Phase 1b may add an explicit
  parent-extra edge.
- `light_side_token`, `c_heavy_side`, `fragment_is_lighter_side` (neutral mass of the atom set versus half the
  precursor's): ALLOWED, semantics stated. An extra's atom set P excludes the deleted atoms, and the
  complement is precursor minus P, so deleted atoms land in the complement, which IS the neutral loss; the
  lighter side is judged by mass of P against the precursor exactly as for ordinary rows. This is not a
  misassignment. Test: SO2 extrusion with P above and below half the precursor mass; `c_heavy_side` counts
  (complement element counts) include the deleted atoms and their hydrogens.
- `c_cut_bonds`, `c_cut_adjacency`, `c_cut_count`: read the boundary rows. Boundary of an extra = precursor
  bonds with exactly one endpoint in P, which includes bonds to deleted atoms (and rule cuts leading to the
  outside), and excludes rule-internal cuts and formed bonds. `c_cut_adjacency` counts pairs of boundary
  bonds sharing an atom split by retained/lost side; two bonds into a deleted S give a lost-side pair, which
  is the extrusion signature the feature was built for. Semantics: "precursor bonds severed to obtain P".
  Test against a hand-computed sulfone and nitro case.
- `_candidate_atom_dropout`: acts on the pooled atom membership of all rows including extras; the complement
  is derived by subtraction so deleted atoms stay in the complement. The `th.rand` draw is shaped by the row
  count, so with dropout > 0 and extras present the random stream of the whole batch differs from the
  control; with rewire OFF it is unchanged. Test: dropout leaves descriptors and masses on the true set.
- Memorization sink (`sink_rows[fragment_batch_indices]`): the extended `fragment_batch_indices` is built
  before the head call, so the sink index follows each extra's own spectrum. Test with a sink model.
- `proposal_selection_only`: required (above); `fragment_log_probabilities` of extras are never used by the
  head.
- Export node merging by atom set: `masster_export.py` (~215-240) already keys nodes on (spectrum, atom-set
  bitmask) and unions rules per node, with `mask_bits = ceil(max_atoms/64)*64`. An extra whose atom set
  equals an ordinary node merges into it and the node carries the union of SMARTS rule names AND rewire
  rule names (two columns). Test with merged rows.

### The after-selection stage

`src/fragnnet/models/masster/rewire_expansion.py`, function `expand_rewire_candidates`, called in
`MASSterModel.forward` after the keep block (`fragment_keep` applied, ~3731-3746, and the post-keep
renormalization at ~3746-3755) and before `pool_component_and_complement_embeddings` (~3758). Pure torch,
integer ops only, no autograd through membership or labels.

Per-fragment tensors that exist at the insertion point, and what the stage does to each (this is the
complete list; "runs unchanged" holds per call, not globally):

| Tensor | Built where | Treatment for extras |
| --- | --- | --- |
| `fragment_log_probabilities`, `base_fragment_log_probabilities`, `retained_*`, `proposal_log_normalizers` | pre-keep and ~3746-3755, from ordinary fragments | NOT extended for the proposal side: proposal outputs, coverage loss and retained normalizer are computed from ordinary rows only. The head is passed `fragment_log_probabilities` (~3871) as an argument; for extras it gets a shape-filler value of 0 and a test asserts it is unread under `proposal_selection_only` (head outputs unchanged when the filler is changed to another finite value) |
| charge log-probabilities | pre-keep (`latent_pathway_charge_log_probabilities`) | Not needed: they enter only the ordinary proposal probabilities |
| `ion_mzs`, `valid_channels` | pre-keep `_fragment_masses_and_valid_channels` (~3614), then sliced by `fragment_keep` | SECOND call of `_fragment_masses_and_valid_channels` on the extras only (their membership, batch indices, cut counts and the valence adjustment below), results concatenated after the sliced ordinary rows. Extras with no valid channel are dropped and counted before the cap |
| `atom_fragment_indices`, `fragment_atom_indices`, `fragment_boundary_bond_indices`, `boundary_fragment_indices` | `_unique_component_memberships` | Extras appended AFTER all ordinary rows; ordinary indices unchanged |
| `fragment_batch_indices`, `fragment_cut_counts`, `fragment_rule_indicators` | pre-keep, sliced | Extended; `fragment_batch_indices` of extras holds the parent's spectrum index. The head's `fragment_indices = arange(num_fragments)`, `num_masks`, `sink_rows[fragment_batch_indices]`, `context[fragment_batch_indices]` and `candidate_inputs.*` are built AFTER the append from the extended tensors, so they stay consistent. Appending breaks the sorted-by-spectrum order of `fragment_batch_indices`: the head argsorts by batch index (`masster_candidate_transformer.py` ~1149) and the other consumers use `index_add`/`scatter_reduce`, but this is a verified-by-test property, not an assumption: a mixed batch with extras of spectrum 0 after ordinary rows of spectrum 1 must give per-spectrum outputs equal to running the spectra separately |
| `fragment_formed_bond_counts` | `None` unless `use_smarts_templates` (~3536, ~3967) | Not overridden. A new per-row signed integer `fragment_valence_adjustment` is passed (see "H channels") |
| `fragment_embeddings`, `complement_embeddings`, `boundary_embeddings`, `candidate_features` | after the stage | Computed over the extended membership by the existing code |

Parity claim, stated precisely (replaces "a0 is bit-equal"): with rewire ON under a0, (i) molecules with
NO matching rule instance are bit-equal in every output, (ii) the proposal outputs
(`proposal_candidate_*`, `proposal_fragment_*`, coverage loss, `bond_logits` gradient) are bit-equal for ALL
molecules, and (iii) the intensity outputs of a matching molecule DIFFER by construction, because extras
join its intensity softmax and a merged row changes the ordinary row's indicator and valence adjustment.
Also with batched molecules, no-match molecules stay bit-equal only if the batch-level ops used are
order-independent per spectrum (segment sums via `index_add`); the test runs a mixed batch.

Stage order of operations (cap last, so the per-precursor cap is applied to DEDUPLICATED, VALID rows; the
earlier draft listed the cap before the edit, which was inconsistent):

1. Hard-error checks at stage entry, in Python, before any tensor work: batch size equals the number of
   packed match groups; for each spectrum the match atom ids are inside `[0, n_atoms)` of that spectrum's
   graph; the atomic number of every motif atom equals `atom_numbers` of the graph atom at that position
   (SMARTS parsed on `mol_smiles` vs the model graph: atom-order mismatch is a hard error, the same rule as
   a SMARTS-vs-CUDA template mismatch); every cut pair resolves to a bond id in `chemical_bond_index`; every
   form pair endpoint is a graph atom; no two ordinary rows of one spectrum share an atom set (the dedup
   precondition); every kept fragment's atoms lie inside its spectrum. A violation is a `ValueError` naming
   the spectrum and rule, never a skip.
2. Pack matches (ragged `match_ptr`, padded `motif_atoms`, `deleted_mask`, `cut_bond_ids`, `form_pairs`,
   atom ids offset by `mol_pyg.ptr`).
3. Containment: pair (F, match) of the same precursor is kept when F contains every motif atom. Pairs
   enumerate in (batch, fragment index, match index) order. One host sync for the pair count; processed in
   chunks of `masster_rewire_pair_chunk` pairs (see Cost) so memory does not scale with pre-cap pairs.
4. Edit: P = F minus deleted atoms. Edges: precursor bonds with both ends in P and not a rule cut, plus form
   pairs with both ends in P. A form pair with a deleted or outside endpoint invalidates that pair; the
   candidate is skipped and counted.
5. Components of the edited graph by min-label propagation (`scatter_reduce` amin plus pointer jumping,
   convergence checked every 4 sweeps, hard cap at the atom count). Deleted atoms are not candidates
   (neutral-loss convention, as templates).
6. Depth-1 continuation (optional): for the top `masster_rewire_continuation_cuts` precursor bonds of P by
   bond logit, recompute components after removing that bond. Formed bonds are never cut in v1. A bond that
   does not split P' yields nothing.
7. Exact dedup on an atom-set bitset of W = ceil(max_atoms/62) int64 words, within the extras and against the
   kept ordinary fragments. Identity is (atom set, edit signature): signature = hash of the rule-cut and
   formed pairs inside the set, 0 for ordinary fragments. A product whose atom set equals a kept ordinary
   fragment adds no row: it sets that ordinary row's rewire indicator and may raise its valence adjustment
   (below). A product with the same atom set but a different edit than another extra stays a separate row.
8. Second `_fragment_masses_and_valid_channels` call on the survivors; drop rows with no valid channel.
9. Cap per precursor (`masster_rewire_max_extra`) on the valid, deduplicated rows. Sort key: parent's
   normalized proposal log-probability (detached; computed from ordinary rows), then depth, then the atom-set
   bitset words as the final tie-break. The key is a function of tensors that both candidate backends emit
   identically, with the bitset as an exact integer tie-break, so the cap does not depend on a float
   reduction order. (An earlier draft said the key is "rounded to float32"; the rounding is only a
   normalizer for equal-valued keys and is not what the parity rests on, the integer tie-break is.)
10. Emit rows appended AFTER all ordinary rows: membership, batch index, boundary bonds, `fragment_cut_counts`
    = number of boundary bonds, `fragment_valence_adjustment`, `rewire_rule_indicators`,
    `rewire_edit_features`, `is_extra`, `parent_fragment`.

### H channels and formed bonds (derivation)

Existing code (`_transferable_hydrogen_shifts`, ~2453-2458):
`valence_gain = sum(free valence slots of the fragment's atoms) + fragment_cut_counts - 2*formed`, then
`max_gain = min(complement_H, valence_gain)` and the ladder uses `max_gain.clamp_min(0)`. The clamp at 0
already exists (so a negative gain is already harmless); a test pins it. The new argument is a signed integer
`fragment_valence_adjustment` (shape per fragment, default None) ADDED to `valence_gain`; the existing
`fragment_formed_bond_counts` path is unchanged (it stays `None` unless SMARTS is on, so it cannot carry
rewire information and a rewire-only model must not depend on it).

Definitions for an extra with atom set P, deleted set D, rule cuts C (precursor bonds), formed bonds Fm:

- Boundary(P) = precursor bonds with exactly one endpoint in P. It depends on P alone. Bonds to D are in
  Boundary(P); a rule cut inside P is not; a rule cut from P to outside is. `fragment_cut_counts = |Boundary(P)|`.
- slots(P) = sum of the PRECURSOR's `atom_free_valence_slots` over atoms of P only. The deleted atoms' own
  valence is not in the sum, so a bond P-D contributes exactly once, through `fragment_cut_counts` (the freed
  slot on the P endpoint). There is therefore no double count between the cut-count term and the deleted
  atoms' valence (the critic's concern was checked: the sum never includes D).
- A rule cut inside P frees one slot on each of its two endpoints (+2, and it is not in Boundary(P)); a formed
  bond inside P consumes one slot on each endpoint (-2).
- Hence `valence_gain(P) = slots(P) + |Boundary(P)| - 2*|Fm in P| + 2*|C inside P|`, i.e.
  `fragment_valence_adjustment = -2*(|Fm in P| - |C inside P|) = -2*net_formed`, net_formed possibly negative.
  A form pair with an endpoint outside P invalidates the pair (step 4), so |Fm in P| is the formed count.

Worked cases (these are the hand-value tests, all pinned):

- Diaryl sulfone Ar-SO2-Ar' -> Ar-Ar' + SO2: P = Ar u Ar', D = {S, O, O}, boundary bonds 2 (Ar-S, Ar'-S),
  slots(P) = 0 (aromatic, saturated), one formed bond Ar-Ar'. gain = 0 + 2 - 2 = 0: a sealed closed-shell
  product, no H gain beyond the ladder's own-H change. `net_formed = 1`.
- Loss-only rule (delete a group, no formed bond, e.g. NO2 loss): gain = slots + (bonds to D) > 0, `net_formed = 0`.
- Internal cut rule (cut a ring bond inside P): `net_formed = -1`, gain = slots + boundary + 2; the ladder
  upper end is raised by 2. Test that a negative net gives a nonnegative ladder bound after the clamp.
- Gain exactly at the clamp: `min(complement_H, gain)` negative -> ladder bound 0, shift 0 still allowed.

Merged rows. Because Boundary(P) and slots(P) depend on the atom set alone, a merged extra and its ordinary
twin have IDENTICAL `fragment_cut_counts`; the earlier worry that the counts differ does not arise. They
differ only through the net-formed term (and the ordinary row's own template formed count `c_t` when SMARTS
is on). The merged row takes the more permissive pathway: `fragment_valence_adjustment_merged = max(adj_ordinary,
adj_extra)` where `adj_ordinary = -2*c_t` (0 with SMARTS off). Since adj_extra < 0 whenever net_formed > 0, the
merge only ever raises the gain when net_formed < 0. A merge counter is logged. Rule indicators and edit
features set on the ordinary row change its input for matching molecules (covered by claim (iii) above).
Skipped: fragments with a nonzero SMARTS template formed count (v1; a disconnected side with a rule on top is
ill-defined); they never produce extras, but still appear as ordinary rows.

Corrections to the code assumptions:

- `smarts_templates.formed_bond_count` is pieces - 1 on the precursor induced subgraph. A rewired product
  that is connected in the precursor but contains a formed edge would get 0. Rewire rows never use it.
- `fragment_formed_bond_counts` is built only on the SMARTS-on path (~3536, ~3967). The stage does not
  touch it; `fragment_valence_adjustment` works with SMARTS off.
- Own-H changes at rearranged atoms (e.g. S=O to O transfer) are not modelled; the learned H ladder absorbs
  them, as for ordinary candidates.

### Featurization, vocabulary, gradients

- `rewire_pathway_residual`: zero-init, bias-free `Linear(MASSTER_NUM_REWIRE_RULES, model_dim)` added after
  `input_projection` in `CandidateIntensityTransformerHead` (`masster_candidate_transformer.py`, mirror of
  `rule_pathway_residual` at ~945-950 and ~1385), through a new optional `rewire_pathway_features` kwarg.
  Built under `fork_rng` only when the flag is on. Buffer `rewire_rule_name_hashes` plus a load pre-hook
  `_restore_rewire_rule_vocabulary` copied from `_restore_smarts_rule_vocabulary` (strict-prefix hash check,
  zero-pad axis 1 of the residual for shorter prefixes, drop newer rules at inference).
- Descriptor `c_rewire_edit` in `masster_candidate_features.py` (`CandidateFeatureSpec` registry,
  `CandidateFeatureInputs.rewire_edit_features`): n_formed, n_internal_cut, n_deleted_heavy, depth; zeros for
  ordinary rows. Resolved only when named in `masster_candidate_features`.
- Connectivity encoder and boundary embeddings do not see formed edges, so a product looks like its
  disconnected pieces and is identified through the token and descriptor. v1 rejects
  `fragment_connectivity_encoder` with rewire on. Phase 1b adds formed edges to `fragment_internal_edges`
  (`masster_component_pooling.py` ~309) with a synthetic bond attribute per (rule, role).
- Gradients, arm `a0` (the only v1 arm): extras are appended AFTER the proposal outputs and retained
  normalizer are computed, so `proposal_candidate_*`, `_proposal_coverage_loss` and `bond_logits` get exactly the control's
  gradient. The intensity loss reaches the shared encoder, the zero-init residual, the descriptor weights and
  the ion-H head. The cap sort key is detached. Limit: a0 gives no signal to WHICH fragment should receive a
  rule; the intensity head must down-weight uninformative extras.
- Arm `a1` (DEFERRED out of v1: it needs extras in the proposal pool outputs, which `proposal_selection_only`
  models define as selection-only, so it is a separate design; kept here as the planned follow-up): extras get a proposal log-probability = the release score of the parent (differentiable through
  bond logits, `_fragment_release_log_probabilities`) + log sigmoid of a rule gate from a new `RewireGateHead`
  (mean-pooled motif-atom embeddings, rule embedding, context; the `SmartsMechanismHead` input pattern; a
  per-(F, match) sigmoid, since the per-mask softmax of `_mechanism_log_probabilities` does not apply). The
  coverage loss sees them. The gate can drift to 1 because the coverage loss has no precision term, so it gets
  the proposal-only weight decay group, a noisy-OR responsibility regulariser as an arm, and NO hard m/z
  label (user rule). a1 renormalises the retained pool over ordinary plus extras, so its proposal outputs
  differ from the control: a separate arm with a coverage sanity check.

## Code changes per file

| File | Change |
| --- | --- |
| `candidates/rewire_rules.py` (new) | Rules, vocabulary, hashes, `RewireMatch`, `build_rewire_matches`, `filter_rewire_matches` |
| `models/masster/rewire_expansion.py` (new) | `expand_rewire_candidates` as above |
| `candidates/builder.py` | ONLY add `rewire_matches: tuple[tuple[RewireMatch, ...], ...] \| None = None` to `MassterStaticGraph` (line ~73). `_build_cpu_candidates`, `_merge_forced_masks` and the SMARTS append block untouched |
| `masster_pl.py` | `_cached_masster_rewire_matches(smiles)` (`lru_cache`, next to `_cached_masster_templates` ~153); fill `rewire_matches` in `_build_static_graph` (~2454) when the flag is on; standalone-inference fallback next to ~2389; hparam guards near ~1586; thread hparams to the model constructor; coverage metrics split into ordinary vs `rewire_extra` next to `_candidate_coverage_by_cut_count` (~163) plus `rewire_survival` and `merge_rate` |
| `masster_model.py` | Constructor args and inline guards gated on `use_rewire_rules`; vocabulary buffer and pre-hook; stage call in `forward`; concatenation of extras into membership, boundary, batch, cut-count and indicator tensors; second `_fragment_masses_and_valid_channels` call for extras; new `fragment_valence_adjustment` argument of `_transferable_hydrogen_shifts` and `_fragment_masses_and_valid_channels` (default None, so existing callers are unchanged); keep proposal outputs before the append |
| `masster_candidate_transformer.py` | `rewire_pathway_residual` and `rewire_pathway_features` kwarg |
| `masster_candidate_features.py` | `c_rewire_edit`, `rewire_edit_features` |
| `masster_export.py`, `inference/export_columns.py` | `pred_node_rewire_rules` plus parent id column, under `return_rule_provenance`; refused for non-MASSter models |
| `config/template.yml` (beside `masster_smarts_templates`, ~836) | New keys below |
| `config/masster/mh_rewire_after_selection_2026_10_14/` (new) | Arm configs |
| `scripts/manual_slurm/fir/submit_masster_mh_ft_memorization_route.sh` or a sibling | Arm name, job name, config and stage maps, usage string |
| `scripts/debug/smarts/` (new) | `debug_masster_rewire_reach_ceiling.py` (G0) and the checkpoint-side expansion script (G0b) |

`scripts/setup_job_cc.py` (not the repo root) needs no change: the rules are compiled in, and the new
`masster_rewire_*` keys are plain scalars, not data paths, so they do not belong in `config_keywords`
(~499-511) or `data_keyword` (~538-560). They are model-gated inline in shared code (read only when
`model_type == masster`) and are NOT read in the batch/data path when the flag is off. A rule-file path would
be a new data-path key and must then be added to both keyword lists.

## Config keys

All default off, each with a one-line first comment (the setup parser reads only the first line), read in
shared code only when `model_type == masster`, no `.get(default)`, no backfill. Checkpoint hparams are the
inference config; the template fills only these new keys for old checkpoints. The rule SET is not a key (it
is compiled in); its identity is the edit-spec hash buffer checked at load.

| Key | Default | Meaning |
| --- | --- | --- |
| `masster_rewire_rules` | `False` | Apply admitted rewire rules to kept fragments; guards in "Required model configuration" |
| `masster_rewire_max_extra` | `64` | Extras per precursor after dedup, validity and cap |
| `masster_rewire_max_matches_per_rule` | `4` | Matches per rule per precursor, deterministic order |
| `masster_rewire_continuation_cuts` | `0` | Depth-1 top-k bond cuts per product; 0 = products only |
| `masster_rewire_pair_chunk` | `65536` | (F, match) pairs processed per stage chunk; bounds memory, not the result |
| `masster_candidate_features` | existing | gains the allowed name `c_rewire_edit` |

(`masster_rewire_proposal_gradient`, the a1 switch, is not a v1 key.)

OFF: no stage, no module, no buffer, no state-dict key, `MassterStaticGraph.rewire_matches is None`, no new
hparam read in the batch path. Every existing output and `MASSTER_RULE_VOCABULARY` hash is bit-identical
(tested). Rewire is independent of `masster_smarts_templates`; template sides that are kept fragments may
themselves receive a rule, but v1 skips fragments with a nonzero template formed count (a disconnected side
with a rule on top is ill-defined); a test runs templates and rewire together.

## CPU/CUDA parity and tests

Parity is a property to TEST, not inherit. The stage runs the same torch code on either device, but its
inputs are the candidate batches of two backends, and `fragment_keep`, the unique-fragment order and the
logits are only as device-independent as those backends make them. (Existing evidence: grouped logits are
bit-equal; the CUDA audit shows template atom/bond order agrees with the model graph and kernel-vs-CPU
forced merges agree over 200 trials, with fp32 summation order a known difference.) So:

- A new test in `tests/test_masster_cuda_full_candidates.py` (cuda marked) asserts that for identical
  molecules the CPU and CUDA candidate batches give the identical unique-fragment ORDER, keep mask, extras
  (atom sets, parents, order after the cap) and public-model forward logits/losses/gradients.
- The cap tie-break is an exact integer (atom-set bitset), so a last-bit difference in a float key can only
  move a row when two keys are within float noise AND differ in the bitset order; the test includes
  symmetric molecules (two equivalent sulfonyl matches) to exercise it, and the stage logs the number of
  float-key near-ties per batch so a parity failure is diagnosable.
- Hard errors at stage entry are Python `ValueError`s (stage step 1), raised before the append and not the
  device assert at `cuda_candidates.py:421`: batch/match count mismatch, motif atom index out of range, motif
  atom element not equal to the graph atom (SMARTS-vs-graph mismatch, same rule as SMARTS-vs-CUDA templates),
  unresolved cut bond id, form pair endpoint missing, duplicate atom sets among ordinary rows, product atom
  outside its parent. Matching is on standardized molecules.

Tests:

- `tests/test_masster_rewire_rules.py` (new): registry counts, append-only vocabulary, edit-spec hashes (a
  rule with a changed SMARTS/edit and the same name changes its hash), SMARTS compile check, match/delete/
  cut/form resolution on hand molecules (diaryl sulfone SO2 extrusion, S-to-O migration, nitro), precursor
  guard, deterministic order and truncation counter, a golden-instance test per rule.
- A `ValueError` test for EACH hard-error path listed above and for each config guard (marginal,
  `connected_fragment_beam`, hybrid backend, connectivity encoder, `intensity_candidate_graph`, missing
  `proposal_selection_only`, `h_transfer_bound != valence`, missing unique mode).
- Oracle: 200 molecules from `molecules.csv`, networkx edit of every kept fragment (fragments from
  `reference_cut_masks.iter_ranked_cut_masks`); compare atom sets, boundary bonds, cut counts, net formed
  counts and parents, including non-splitting cycle bonds, deleted atoms, multi-component P and a form pair
  with an outside endpoint.
- Coverage equivalence on small molecules: with an unlimited pool, after-selection products plus the ordinary
  pool equal the gate's reachable sets restricted to products whose motif bonds are uncut (documents the
  straddling-cut exception); retained-side atom sets match the gate.
- Dedup: an extra equal to a kept ordinary fragment adds no row and sets only the indicator; same product
  from two parents merges; same atom set with different edit signatures stays separate; merge counter.
- Valence (hand values from "H channels"): SO2 extrusion gain 0; loss-only rule; internal-cut rule with
  negative net formed and the ladder clamp at 0; merged row takes the maximum adjustment; adjustment passes
  through `_fragment_masses_and_valid_channels` for the extras call.
- Placement: output unchanged when the extras' filler `fragment_log_probabilities` changes; mixed batch with
  extras of spectrum 0 after ordinary rows of spectrum 1 equals per-spectrum runs; `sink_rows`,
  `context[fragment_batch_indices]` and `fragment_indices` consistent after the append.
- Candidate-set features: `c_heavy_side`/`light_side_token` (P above and below half mass), `c_cut_bonds`,
  `c_cut_adjacency` with bonds to deleted atoms, `_candidate_atom_dropout` keeps true-set descriptors.
- Bitset width: molecules at and around `W` boundaries (61, 62, 63, 124, 125 heavy atoms; `W = ceil(max_atoms/62)`)
  and the largest molecule in the data; stage and export `mask_bits` agree.
- Mixtures and grouped CE views: with `share_proposal_across_ev` (shared logits per `proposal_group_ids`) the
  extras are identical across the views of one molecule; without sharing each spectrum's pool gets its own
  extras and a test confirms no cross-spectrum leakage.
- Multi-adduct: a batch with several `allowed_precursor_types` and a rule restricted to `[M+H]+` only fires
  on the allowed spectra.
- Coexistence: SMARTS templates ON and rewire ON together: template-side rows with a nonzero formed count
  get no extras, other rows do, outputs finite, no index collision between the SMARTS append and the stage.
- Annotation export under rewire: `pred_node_rewire_rules` with merged rows (an extra and an ordinary
  fragment with the same atom set give one node carrying both rule sets), parent id column, non-MASSter
  refusal (`tests/test_inference_masster_annotation.py`).
- Feature-off bit-identity (pattern of `tests/test_masster_smarts_pathway.py`): state-dict keys, forward
  outputs, loss, vocabulary hash.
- Gradient isolation (pattern of `tests/test_masster_pl.py:304`): in a0 the proposal outputs and gradients
  are bit-equal with rewire on and off for ALL molecules, a no-match molecule is bit-equal in the intensity
  outputs, and the intensity gradient reaches the residual and descriptor weights.
- Checkpoint: strict-prefix rewire vocabulary loads with zero padding; reordered, renamed or edit-changed
  raises; an old checkpoint loads with the feature off and errors clearly if switched on without the buffer.
- Setup parsing of the new keys.

## Cost estimate

Pair-count figures are estimates and are explicitly UNMEASURED; G0b measures them on the actual pool.
Fraction of single-bond-cut fragments that fully contain a given match: sulfonyl 0.30, nitro 0.41, P 0.52,
halogen 0.59. With the 256-fragment unique target that is roughly 80-150 (F, match) pairs per matching rule
instance. Extras are bounded by the cap: at most 64 per precursor, +25% of 256 for a matching molecule, +0
for others; with 10-30% of molecules matching an admitted rule union the mean is +2% to +8% nodes (an
assumption chain, not a measurement). The mean is not the binding statistic: the transformer cost per
candidate is linear in candidates per spectrum (quadratic with `candidate_graph`, forbidden here) and the
batch is padded by its largest spectrum, so G0b reports p50/p95/max nodes per spectrum and the
tail-molecule values.

The cap is applied AFTER dedup and validity, so it does not bound the stage's work or memory. Pre-cap pairs
per precursor <= (kept fragments, 256) x (matches M <= rules x `max_matches_per_rule`), times (1 + k) with
depth-1. Example: M = 20, k = 0, 512 spectra per training batch: 256 x 20 x 512 = 2.6M pairs; as a bool
membership matrix with Amax = 100 atoms that is 262 MB (int64 2.1 GB), as a W = 2 word bitset 42 MB. The
earlier "under 5 MB" figure assumed the capped 64 rows and is withdrawn. The stage therefore processes pairs
in chunks of `masster_rewire_pair_chunk` (bitsets, not dense matrices, between chunks); peak memory is
chunk x Amax bytes plus the survivors, independent of the pre-cap count. Measure peak stage memory and
host syncs (the pair count, and one per chunk loop) in G0b.

CPU precompute (`build_rewire_matches` per uncached SMILES, cached like the template cache): estimated 10-30
ms per uncached SMILES for 10 rules; gate it with a wall-clock measurement over the full train+val SMILES
set (29,802 molecules here; the NIST20+MSGYM training pool is larger), reported as total seconds and p95 ms
per SMILES. Wall-clock gate (proposed bar, no precedent): precompute total below 5% of one epoch's wall time of the control, otherwise
cache to disk keyed by SMILES and rule-spec hash.

Cost scales with NODE count, so headline cost is wall-clock per epoch and candidates per spectrum against
the control (best-by-hour first).

## Experiments

### Phase 0 (CPU, no training)

- **P-1 tail seed spread (before any threshold is fixed).** Score the existing `i1`-recipe replicates (and
  `i3`/`i4` if the per-spectrum predictions exist) on the 669-spectrum / 194-molecule validation tail,
  report the tail-mean `cos_hun_inst_tol` per seed, the seed-to-seed SD, and the paired-bootstrap SE of a
  difference over the 194 molecules. The whole-split spread (0.0042) does not transfer: the tail is small
  and heavy-tailed, and its spread is expected to be several times larger. If per-spectrum predictions of
  at least two seeds are not available, two more `i1` seeds are a prerequisite (they are in the Phase 1
  replicate budget anyway). Every tail threshold below is `max(stated floor, 2 x measured tail SE)`.
- **G0 reach ceiling** (`debug_masster_rewire_reach_ceiling.py`, validation split of `rearrangement.pkl`):
  per rule and overall, share of rearrangement-peak intensity reachable at depth 0, 1 and 2 (10 ppm, +-4 H as
  mining) in three variants: (R_all) the rule applied to every connected set of G (unlimited pool),
  (R_after) applied to the i1 256-unique pool, (R_sel) top-256 by ranking over the rewired graph (checkpoint
  bond logits, CPU on the validation tail and a matched non-tail sample), each versus the null-matched
  control (`null_smarts`), versus the t1i precedent, plus matches per molecule and extras per matching
  molecule. Report the validation tail size first (669 spectra, 194 molecules, above).
  Basis of the bars (arithmetic with an assumption, not a measurement): inside the validation tail the
  rearrangement peaks carry a mean 0.405 of the intensity. If the surviving rules reach a fraction r of
  that intensity, the L1-intensity ceiling of the tail gain is about 0.405 r (cosine is not L1, so this is
  an order-of-magnitude bound). The model will convert only part of the reach; the t1i G1 precedent
  (+0.196% of bearing-spectra intensity reach) shows reach converts poorly, and a conversion fraction c is
  unmeasured. With an assumed c = 0.25: r = 10% gives 0.405 x 0.10 x 0.25 = +0.010 (the tail threshold
  floor), r = 3% gives +0.003, below the whole-split seed spread. Hence **pass: r(R_after, depth 1) >= 10%**
  of rearrangement intensity, each kept rule beats its null (CI over molecules excludes 0) and covers >= 1% of
  tail intensity; **kill: r < 3%** or depth-1 reach not above null. Between 3% and 10%: stop unless
  R_sel exceeds R_after by a margin that justifies Phase 2. These bars are revisited if P-1 shows a tail SE
  above 0.005 (then the +0.010 floor is not detectable) or if the user supplies a different c.
- **G0b checkpoint-side expansion**: run the stage on the existing `i1` checkpoint, val splits
  (`nist20_inchikey`, MassSpecGym), CPU, logging per rule pairs, extras after dedup/validity/cap, merge rate,
  nodes per spectrum (mean, p50, p95, max) and node growth, intensity-weighted recall of the tail peaks, stage
  wall-clock and peak memory, and F SURVIVAL: for each rearrangement-peak product S that R_all reaches, the
  set of minimal parents F (connected, contains the motif, S a component of the edited F under at most
  `depth` cuts); survival = share of (peak, intensity-weighted) whose product has at least one parent in the
  kept 256-unique pool. Pass: node growth mean < +10% AND p95 < +25% (the cap's own bound for matching
  molecules) AND stage precompute+forward wall-clock inside the budget in "Cost estimate", AND R_after >= 50%
  of R_all. Trigger for Phase 2: F survival < 70% (the proposed bar; R_after/R_all below 50% with survival
  above 70% means continuation depth, not selection, is the limit and is a Phase 1 knob instead). G0b can
  only under-reach G0 (a subset by construction), so the ratio alone cannot separate survival from
  continuation; survival is measured directly.

### Decisions that gate Phase 1

Each has a default, so Phase 1 may start on the defaults if the user does not answer; answers change the
arms and the checkpoint format, so they are listed here, not buried in the questions at the end.

| # | Decision | Default for Phase 1 | Changes if answered otherwise |
| --- | --- | --- | --- |
| 1 | Extras displace ordinary fragments (equal candidate count) or add | ADD (cap 64), reported as nodes per spectrum | A displace arm needs a fixed per-spectrum node budget and a rule for which ordinary rows go; adds a control arm |
| 2 | a0 intensity-only versus a1 proposal gradient | a0 only (a1 is incompatible with `proposal_selection_only` as built) | a1 needs its own design before it is a Phase 1 arm |
| 4 | Separate rewire vocabulary and residual versus appending to `MASSTER_RULE_VOCABULARY` | Separate (checkpoint-safe) | Appending changes `MASSTER_NUM_RULES` and the shapes/init of `rule_embedding` for every SMARTS model: a different checkpoint compatibility contract |

Questions 3, 5, 6 and 7 (rule set, formed-bond cuts, Phase 2 gate form, tail definition) do not block
Phase 1; 3 is answered by G0.

### Phase 1 (Fir, i1 recipe, 64 epochs, seed 397623 for screens, `upload_checkpoints`)

| Arm | Content |
| --- | --- |
| `i1` | control (existing `i50ig7rf` lineage; rerun only if branch code changed numerics) |
| `a0` | rewire on, depth 0, intensity-only extras |
| `a0d1` | a0 + `masster_rewire_continuation_cuts: 4` |
| `a0null` | a0 with null-matched edits (same pool size and rule columns, wrong chemistry) |

Replicate `i1`, the best arm and its null with 3 seeds each (whole-split spread 0.0042; tail spread from P-1;
paired by molecule). Add Enveda-180 only after a pass (rules derive from NIST20 and MassSpecGym).

Metrics, all on validation: whole-split `cos_hun_inst_tol` (independent-checkpoint selector, as in the i1
worst-prediction work); the 669-spectrum / 194-molecule tail (rearrangement share >= 25% primary, >= 10%
secondary) with a paired bootstrap over MOLECULES (2000 reps) of arm minus control; spectra whose molecule
has no match (must be bit-equal under a0); rearrangement-peak pool recall and extra precision by rule and NCE
bucket; node count mean and p95; rule-fragment survival and merge rate. Wall-clock first: best-by-hour, then
epoch-matched, plus step time and candidates per spectrum. Expectation: whole-split movement below seed
noise (0.011 x tail delta).

Gates (T = tail threshold = `max(0.01, 2 x tail SE from P-1)`): promote when tail delta >= +T with 95% CI > 0,
tail beats `a0null` by >= T/2, whole-split delta >= -0.0042, non-rearrangement spectra delta >= -0.003, nodes
per spectrum p95 <= 1.25x control, and wall-clock per epoch <= 1.3x control.
Kill: (a) pool recall gain on the tail < +1 point (no headroom; matches the t1i G1 stop); (b) tail CI includes
0 at 3 seeds while pool recall gain > +2 points (model cannot use the candidates; route to token and prior
work only); (c) `a0null` equals `a0` within the tail SE (gain is candidate count, not chemistry); (d)
whole-split drop > 0.0042 or wall-clock > 1.5x with no tail gain, or step time +15% for a tail gain < T.

### Phase 2 (only on the G0b trigger)

In-selection: per match build a virtual molecule whose eligible bonds exclude T (rule cuts plus every bond to
a deleted atom), call `top_mixture_bond_sets_raw` / `rank_batch_cuda` on that segment, map ids back and union
with T, then topology on the edited endpoint list (formed edges appended as always-intact bonds, a
`bond_global_map` with -1 for formed). Layout `[ordinary, SMARTS, REWIRE]` with new batch fields. Unique key
(atom set, edit signature). A rule gate replaces the Bernoulli charge on T (unnormalized; see open
questions). Verify first whether `rank_batch_cuda` can drop forced singles per call (`cuda_mask_ranking.py`
`_extra_forced_words`, `_prepare_layout`, kernel base-forced rows); a wrapper change may be needed. Budget
semantics differ from `masster_ranked_mask_budget`. Plan separately if triggered.

## Risks

- Pool truncation: F must be kept, and the ordinary ranking can drop a multi-cut F (see phasing). 72% of tail
  intensity is in 30-70%-heavy-atom ions that need F then rule then cuts. G0b measures F survival directly.
- a0 cannot choose which F gets a rule; at the cap the intensity softmax is diluted. Mitigations: a1 gate,
  tighter cap, per-rule caps, the null arm. The a1 gate is deferred (needs a proposal-side design).
- Atom-set identity and the merge policy (maximum valence adjustment). Boundary and slots depend on the atom
  set alone, so only the net-formed term differs between a merged extra and its ordinary twin.
- Rule drift: a rule edited under the same name is caught by the edit-spec hash at checkpoint load.
- Net formed counts can be negative; wrong slot semantics silently invalidate channels. Watch the
  `valid_fragments` rate and the dropped-no-valid-channel count; the hand-value tests are mandatory.
- Formed bonds uncuttable in v1 (ring re-opening products missed); depth-1 limited to precursor bonds.
- Silent absence: fragment-level selection or the cap can drop a rule product. Report survival; never skip
  silently on a missing bond id or mask.
- Unmeasured pool cost; wall-clock and NODE count rise at inference too, so ICEBERG/GLACIER comparisons stay
  fair only if rewire is on for MASSter alone and node counts are reported.
- Rule quality: only gate-passing rules; rules from [M+H]+ Orbitrap mining have no evidence for other
  adducts or CE, so guard by precursor type.
- Pitfall of the forced-mask variant if ever adopted: sides must be judged connected on the induced subgraph
  of G, otherwise a side that is only split by an internal cut collides with an ordinary atom set.
- Shared files (`masster_model.py`, `masster_pl.py`, `template.yml`) are edited live by other sessions: new
  code goes in new files, shared edits are small and flag-gated, stage explicit paths only.

## Open questions for the user

Questions 1, 2 and 4 are mirrored with defaults in "Decisions that gate Phase 1"; Phase 1 is gated on them
only in the sense that the defaults apply unless the user answers.

1. Must the total candidate count equal the control's (extras displace the lowest-scoring ordinary
   fragments), or may extras add to it? Default: add (cap 64). It changes the cost story and the controls.
2. Is intensity-only a0 acceptable as the first read, given it gives rule selection no signal? Default: yes;
   a1 is deferred, not an arm.
3. Which rules enter v1: only gate survivors with real > null (default; G0 answers it), or all mined families
   (S+O, N+O, halogen, O-only, P)? May the 8 commented-out SMARTS rules be revisited?
4. Separate rewire vocabulary and residual (default, protects SMARTS checkpoints), or append to
   `MASSTER_RULE_VOCABULARY` so `SmartsMechanismHead` and the p1 token are shared?
5. Is restricting continuation cuts to precursor bonds acceptable, or are formed-bond cuts needed (virtual
   bond ids)?
6. If Phase 2 runs: may the rule gate replace the Bernoulli charge on T (unnormalized), or do you want a
   per-molecule normalization with an "ordinary" option?
7. Confirm the tail definition (>= 25% rearrangement intensity, validation only: 669 spectra, 194 molecules)
   and that `nist20_inchikey` validation, not only ORB/ORB_F, is the evaluation target.
8. After selection versus as part of selection: the plan starts after selection because it needs no builder or
   kernel change and its cost is a bounded stage (a cost ordering); both remain open, and Phase 2 is the
   in-selection option.

## Documentation steps (`docs/rules/documentation.md`; not done by this edit, which touches only this file)

- Add a row to the MASSter generalization hub's "All documents" table
  (`docs/workstreams/masster_generalization.md`, near the SMARTS rows ~101): `[Rewire rules in training](../plans/masster/masster_rewire_rules_training_plan.md) | PROPOSED (2026-10-14) | Mechanism rules with continuation as a training stage; G0/G0b gates`, plus a one-line pointer in the hub's SMARTS section.
- Update the hub's "Where it stands"/"Open threads" and replace the MASSter generalization block in
  `docs/project-state.md` (no dated append); a `grep -i rewire` in the hub, `docs/project-state.md` and
  `docs/README.md` finds nothing today.
- Write the finding `docs/findings/masster/masster_rearrangement_mining_2026_10_14.md` (mining table, the
  validation-tail counts, later the G0/G0b cost numbers) and link it from here; this plan then keeps only
  conclusions (one canonical location per fact).
- Fill "Admitted rules" from G0, then move the status to ACTIVE when Phase 1 is scheduled.
