# Fragment representation: bond sets, and where the charge sits

**Status 2026-07-22: DESIGN PROPOSAL. Nothing here is implemented or measured.** The motivating
measurements are cited inline; everything else is reasoning to be tested, and §6 lists the gates
that come first.

## 1. What we do today, and why change it

A DAG node (fragment) is represented by `cc` — a 128-bit atom bitmask — and its embedding comes from
pooling contextualized atom features over that mask.

Two measured facts motivate revisiting this:

- **Per-node cost is the wall.** The bond-breakability depth-extension line was STOPPED 2026-07-03
  because the cost driver is nodeMLP (per-node), not edges: masking gives 30–39% edge reduction but
  only 1.5–6.6% node reduction, since redundant multi-parent routes to the same fragment absorb it
  (3.00 edges/node avg). See `docs/bond_breakability_mask_plan.md`. Any real D4/D5 extension needs
  cheaper nodes, not fewer edges.
- **The representation is load-bearing.** The cc-corruption probe drops cos 0.82 → 0.61 under mask
  randomization (`docs/cc_corruption_probe_finding.md`), so whatever replaces cc has to carry
  comparable information.

## 2. The proposal

Describe a fragment as a **set of bonds** rather than a set of atoms:

```
fragment = { directed boundary bonds:  f(h_kept, h_lost, bond_feats) }   # cleavage chemistry
         ∪ { ionizable sites: pooled h_a over heteroatoms only }         # charge chemistry
         + formula / composition features                                # global composition
```

pooled permutation-invariantly (DeepSets machinery already exists in the readout).

### 2.1 Radius 2, learned by the existing GNN

Bond environments are **not** a discrete vocabulary. A k-layer GNN over the molecule graph already
produces, at each atom, an embedding of its radius-k neighborhood, so the radius-2 bond environment
*is* `(h_a, h_b)` from a 2-layer pass. No hashing, no embedding table, no thin-support buckets —
unseen environments land near chemically similar ones by construction.

**Radius 2, not 1**, for a concrete chemical reason: `atom_env(a, radius=1)` gives a's identity, its
neighbours' identities and the bond types to them, but *not* the neighbours' other bonds. So
alpha-to-carbonyl, benzylic and allylic activation — the motifs that govern where a molecule cleaves
— are invisible at r=1. Independent evidence that 2 bonds is the right scope: the hand-built coarse
descriptor in `debug_dag_bond_descriptor_stats.py` carries an explicit `alpha_hetero` flag, i.e. a
radius-2 feature bolted onto an otherwise radius-1 descriptor.

Note the counter-evidence and why it is weak: `scripts/eval/flux_beta_bucket_variance.py` found ICC
saturating at r1 (0.7177 vs 0.7219 at r2). That measures how well bond environment explains the
*trained flow model's* β — a β that was never anchored to truth and learned nothing about branching.
It does not show r=2 lacks real chemical signal. The discriminating test is the oracle reference
(§6.3).

### 2.2 Boundary-only vs all bonds — two different bets

| variant | cost | buys |
|---|---|---|
| all bonds | O(#bonds) ≈ O(#atoms), same order as cc pooling today | representation quality, no cost relief |
| **boundary-only** | O(1–3) per node | the nodeMLP cost win → depth extension becomes affordable |

Boundary-only is less lossy than it sounds: with a depth-2 GNN each boundary endpoint embedding
already summarises 2 bonds *into* the fragment interior, and the formula carries global composition.
For small fragments it is nearly complete; for large ones it captures the cleavage neighbourhood and
delegates size/composition to the formula.

### 2.3 Directed boundary pairs

For each cut bond, which endpoint is retained and which is lost is known without any pooling.
`(h_kept, h_lost, bond_feats)` is strictly more informative than an unordered pair and free.

### 2.4 Known caveat: parent-context contamination

Atom embeddings are computed on the **intact** molecule graph (FraGNNet runs the mol GNN once;
fragments index into it), so `h_a` includes context from atoms on the far side of the cut — the
fragment "sees" material it does not contain. Re-running the GNN per fragment is prohibitive.

Cheapest mitigation: pass the `is_boundary` flag and the directed pair and let the node MLP learn to
discount it. Stronger: gate messages across cut bonds, but that requires per-fragment passes.

**This is not a regression** — cc pooling uses the same parent-context atom embeddings today. It is
an inherited limitation, not one the change introduces.

## 3. Where the charge sits

This is the weak point of a boundary-only description and needs explicit handling.

Charge retention is governed by ionization site: proton affinity for `[M+H]+`, acidity for `[M-H]-`,
chelation for `[M+Na]+`. Those sites are usually **interior** heteroatoms — a basic amine, a pyridine
N, a carbonyl. A boundary-only view sees 2 bonds in from the cut and nothing else, and the formula
says there are two nitrogens but not whether they are basic amines or amides. So the representation
that is cheapest for cleavage chemistry is exactly the one blind to ionization chemistry.

**Fix: pool over heteroatoms only.** Typical fragments have 2–6, so the cost win over full-cc pooling
survives while restoring precisely the atoms that determine where charge goes. The GNN embeddings
already encode aliphatic-basic vs amide vs aromatic N — that is what a 2-layer pass is for.

### 3.1 Adduct conditioning

The charge carrier changes which site wins, so site scoring is conditioned on `prec_type` (already an
embedding, `src/fragnnet/model/base_model.py:258`): score each ionizable site with the adduct
embedding as context, then aggregate (softmax / logsumexp) into a fragment-level "can this hold the
charge under this adduct" term.

Sanity check with real content: `[M+H]+` and `[M-H]-` should learn near-opposite site preferences
from the same atom embeddings. If they don't, the mechanism is fitting noise rather than chemistry.

## 4. The supervision opportunity: complementary pairs

**Status: gates 6.1 and 6.2 MEASURED (2026-07-22). Survives, but not in the binary form originally
written here.**

Cutting one bond yields two pieces; typically only one is observed, because only one keeps the
charge. Both sides exist as DAG nodes (§6.1), so each cut defines a competition — which side held
the charge.

Why this is attractive: unlike sibling branching, the pair is defined by a *single cut*, so it does
not suffer the multi-parent entanglement that killed edge-β supervision
(`project_beta_identifiability_pruning`, `project_fluxmpnn_stage3`). Pair membership is recoverable
offline with no extra bookkeeping — `cc[a] ∩ cc[b] = ∅ ∧ cc[a] ∪ cc[b] = cc[parent]` identifies the
cut uniquely, and it determines the parent.

### 4.1 What the measurement says

From §6.2: 1,326 decidable pairs per spectrum covering 70.9% of spectrum intensity — far more reach
than the sibling-ranking targets (0.285). The label is **not** a mass proxy: the larger side wins only
55.01% of the time, near chance, so whatever decides it is chemical rather than size-driven. Those two
results are why this line stays open.

### 4.2 Why it must be soft, not binary

Only **3.37%** of decidable pairs have a winner explaining a peak that no other DAG node claims. In
the other 96.6% the evidence is a shared attribution across many nodes at the same m/z. A hard "this
side won" target is therefore exactly the m/z-only many-to-one per-candidate label ruled out by
`feedback_candidate_level_hard_labels`, and 3.37% is the number that says so.

The form to build is responsibility-weighted, mirroring the soft sibling loss in
`docs/sibling_ranking_target_plan.md`: a pair contributes in proportion to how exclusively its
winner's peaks are attributable to it, rather than asserting a winner outright. The 1,326 pairs/spectrum
is **not** 1,326 independent facts; the effective count is much lower and is not yet quantified.

Note also that a winner is inferred partly from the *absence* of a peak for the partner, and absence
has causes other than charge retention — below detection, aliased onto another node's m/z, or the
fragment not forming at all. This is a second reason the target must carry uncertainty rather than
commit.

## 5. What this changes elsewhere

- If boundary-only holds up, node representation shrinks enough that D4/D5 depth extension — stopped
  on node cost — becomes affordable. That attacks coverage, cost, and the CE-dependent recall gap at
  once, and is a larger prize than the branching work.
- This is the reverted `BoundaryBondEncoder` idea (removed 2026-07-13) but as a **replacement** rather
  than an addition, with the cost argument that was not on the table then, plus directed pairs and the
  charge-site term. The earlier priority call does not bind.
- It does not address the measured bottleneck (generalization, not capacity —
  `project_capacity_vs_generalization`); the cost argument is what carries it.

## 6. Gates — run before building anything

1. **Do DAGs retain both complementary fragments as nodes?** — **PASSED, 2026-07-22.**
   Both `compute_frags.pyx:358` and `multi_cut_bfs.pyx:687` register both components of a cut. The only
   thing that would drop a side is `min_frag_atoms`, which is **0 on all production DAGs**
   (`preproc_scripts/03_prepare_dag_feats.py:818` default; `frag_gen_nist20_d3_mbfs*.sh`). Measured on
   300 mols / 313,558 nodes: 100.00% of parent-child slots are half of a complementary pair;
   1,399,558 pairs, median ~20 per parent, tail to 875.

   This gate could not really have failed once `min_frag_atoms=0` was confirmed — both CCs are emitted
   by construction. Treat it as a precondition check, not evidence for the design.

   *Caveat:* `min_frag_atoms` is a config knob, not an invariant. A target builder relying on pairing
   should assert it rather than assume it; a regeneration with a nonzero value would silently break
   pairing for exactly the small-fragment cases where charge retention is most lopsided.

2. **How often is exactly one member observed?** — **MEASURED, 2026-07-22.** 100 train mols / 1,988
   spectra:

   | | |
   |---|---|
   | pair-spectra evaluated | 15,269,172 (7,681/spectrum) |
   | neither observed | 80.71% |
   | both observed (ambiguous) | 2.02% |
   | **exactly one observed** | **17.26%** → 1,326/spectrum |
   | winner has a private peak | 3.37% → 44.7/spectrum |
   | larger side wins | 55.01% |
   | intensity covered by winners | 0.709 |

   Verdict: passes on volume and on non-triviality (55% ≈ chance rules out a mass proxy), fails on
   attribution cleanliness (3.37%). Drives the soft-form rewrite in §4.2.

   *Two probe bugs found and fixed while running this, recorded so they are not repeated:* an
   alias check asking whether a peak is claimed by both members is **vacuous** — if exactly one side is
   observed the partner explains no peaks and cannot co-claim one, so it returns 100% by construction;
   the meaningful test is whether the winner's peaks are claimed by *any other DAG node*. And coverage
   must be the union of peaks explained by winning nodes, not a sum of per-node intensities — winners
   share peaks, and the naive sum exceeded 1.0.

3. **Oracle reference for radius.** ICC of *observed* `q_obs` by bond environment at r1 vs r2. Settles
   whether r=2 carries real branching signal beyond r=1, which §2.1's ICC number does not. Shares
   machinery with the sibling-ranking target builder.
4. **Representation probe.** Using the 583k sibling targets, predict observed sibling ordering from
   (a) cc-pooled atom embeddings vs (b) directed boundary-bond embeddings + heteroatom pool + formula.
   Same targets, two feature sets, offline. Measures exactly the trade being proposed. **This is the
   gate that tests the load-bearing claim** — gates 1 and 2 concern §4, which is the speculative part;
   the cost case in §1–§3 is what the proposal actually rests on.

All four are offline and need no training run.
