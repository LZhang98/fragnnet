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

Cutting one bond yields two pieces; typically only one is observed, because only one keeps the
charge. **If both sides exist as DAG nodes**, each cut defines a binary competition — which side held
the charge.

Why this is attractive: unlike sibling branching, the pair is defined by a *single cut*, so it does
not suffer the multi-parent entanglement that killed edge-β supervision
(`project_beta_identifiability_pruning`, `project_fluxmpnn_stage3`). The two nodes are distinguishable
in the spectrum by m/z, and the competition is local and identifiable.

This would be a new supervision signal, orthogonal to the marginal sibling-ranking targets in
`docs/sibling_ranking_target_plan.md`, and probably cleaner.

**It is entirely conditional on the gate in §6.1.**

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

1. **Do DAGs retain both complementary fragments as nodes?** If BFS keeps only the retained side, §4
   collapses entirely. Check `compute_frags.pyx` and a handful of real DAGs. *Everything in §4 depends
   on this.*
2. **If yes: how often is exactly one member of a complementary pair observed?** That frequency is the
   headroom for a charge-retention head — same shape as the sibling coverage gate, and it either
   justifies the work or closes it.
3. **Oracle reference for radius.** ICC of *observed* `q_obs` by bond environment at r1 vs r2. Settles
   whether r=2 carries real branching signal beyond r=1, which §2.1's ICC number does not. Shares
   machinery with the sibling-ranking target builder.
4. **Representation probe.** Using the 583k sibling targets, predict observed sibling ordering from
   (a) cc-pooled atom embeddings vs (b) directed boundary-bond embeddings + heteroatom pool + formula.
   Same targets, two feature sets, offline. Measures exactly the trade being proposed.

All four are offline and need no training run.
