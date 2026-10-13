# Fragment representation: bond sets, and where the charge sits

> **Status:** DEFERRED · **Updated:** 2026-10-06
> All four gates measured, objective corrected to regularization (read §7 first). The never-run `cc_atom_dropout_p{00,01,02}` sweep was the next step; no result is recorded in the docs and no work has followed since 2026-07-23.

**Status 2026-07-23: ALL FOUR GATES MEASURED, THEN THE OBJECTIVE WAS CORRECTED. Read §7 first.**

§1–§6 were written against a *cost* motivation. That motivation is wrong (and §2.2 refutes it on its
own terms). The real goal is (1) focus on the **cut**, (2) reduce **fragment identifiability** to cut
overfitting. §7 measures that objective and reaches a different — and better — answer than §6.4 did.
The summary below is retained because the measurements stand; only their relevance changed.

- **§2.2 cost argument — REFUTED.** Per-node cost is dominated by `NodeMLP`, which is node-count-bound
  and indifferent to the readout. The bond-set readout measures **+109%** at faithful width and ~−4%
  at best under aggressive narrowing. It is not a route to cheaper nodes, and cheaper nodes were the
  stated motivation (§1).
- **§6.4 representation argument — TRUE BUT ALREADY BANKED.** Cut-bond information genuinely beats the
  cc atom mask, and once cut bonds are present the mask adds ~nothing. But production already carries
  `boundary_pair`; measured against *that* baseline the proposal gains ~2σ on TV and ~1σ on
  dominant-sibling mismatch.
- **§4 complementary-pair supervision — still open**, in soft form only (§4.2), and never depended on
  the cost argument.

**Under the corrected objective (§7), the live design is different and better-supported:**

| term | use | evidence |
|---|---|---|
| cut | static r2 bond bucket (untested) — else **k=1** GNN bond encoder | §7.3b: k=1 dominates k=2 on every axis; dom_mis non-monotonic 0.489 / **0.450** / 0.464 |
| charge | lossy heteroatom descriptor, tolerates k=1–2 | §7.2: best ranking of any arm (0.405, ~7σ over cc) |
| formula | **omit** | §7.1: re-identifies the fragment, 29% → 73% singletons |

Cheapest production step, zero new code: run the already-written, never-run `cc_atom_dropout`
sweep (§7.3). The `boundary_pair` → `cut_bond` swap noted under §6.4 is still valid but is a
small effect and is *not* the main line.

## 1. What we do today, and why change it

A DAG node (fragment) is represented by `cc` — a 128-bit atom bitmask — and its embedding comes from
pooling contextualized atom features over that mask.

Two measured facts motivate revisiting this:

- **Per-node cost is the wall.** The bond-breakability depth-extension line was STOPPED 2026-07-03
  because the cost driver is nodeMLP (per-node), not edges: masking gives 30–39% edge reduction but
  only 1.5–6.6% node reduction, since redundant multi-parent routes to the same fragment absorb it
  (3.00 edges/node avg). See `docs/plans/fragnnet/bond_breakability_mask_plan.md`. Any real D4/D5 extension needs
  cheaper nodes, not fewer edges.
- **The representation is load-bearing.** The cc-corruption probe drops cos 0.82 → 0.61 under mask
  corruption (`docs/findings/fragnnet/cc_corruption_probe_finding.md`; magnitude unverified since 2026-10-12, the corruption leaked atoms across molecules), so whatever replaces cc has to carry
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

**The earlier counter-evidence has been retired by measurement.**
`scripts/eval/flux_beta_bucket_variance.py` found ICC saturating at r1 (0.7177 vs 0.7219 at r2), but
that measured how well bond environment explains the *trained flow model's* β, not truth. Gate §6.3
ran the same decomposition against *observed* branching and found no saturation: r1 → r2 adds +42%
relative (0.0936 → 0.1331 raw, 0.1277 → 0.1817 noise-corrected). r=2 carries real chemical signal.

### 2.2 Boundary-only vs all bonds — the cost claim, and why it FAILS

**Status 2026-07-23: MEASURED and REFUTED.** This section originally read:

| variant | cost | buys |
|---|---|---|
| all bonds | O(#bonds) ≈ O(#atoms), same order as cc pooling today | representation quality, no cost relief |
| **boundary-only** | O(1–3) per node | the nodeMLP cost win → depth extension becomes affordable |

Both the width claim and the nodeMLP claim are wrong.

**The width is not O(1–3).** Measured on the gate-4 cache (25 mols, 3,404 fragments that appear in a
sibling group): cut bonds per fragment mean **5.32** (median 6, max 6 — the d3 multi-cut ceiling, not
a cap), heteroatoms in the fragment mean **3.54**. So the proposal pools **8.86** items per node
against cc's **15.40** — a 42% reduction, not an order of magnitude, and on **16%** of nodes the
bond-set readout is *wider* than the atom mask it replaces.

**There is no nodeMLP win, because nodeMLP does not depend on the representation at all.**
`NodeMLP` (`nn_blocks.py:841`) is `num_layers × (Linear(h,h), ReLU, Linear(h,h))` applied per node:
its cost is set by node **count** and `frag_hidden_size`, and the readout only feeds the single
`input_project` layer ahead of it. Changing how each node's vector is pooled cannot move it. The one
thing that reduces nodeMLP cost is *fewer nodes* — which is exactly what
[bond breakability mask plan](bond_breakability_mask_plan.md) targeted and exactly why that line stopped when node reduction
came in at 1.5–6.6%.

**Measured, production scale** (RTX 2070S, 34k nodes, `mol_hidden=384`, `frag_hidden=256×4`):

| component | ms | note |
|---|---|---|
| cc pooling (arm A) | 9.24 | 54% of per-node cost — pooling IS significant |
| NodeMLP (4 layers) | 6.32 | node-count-bound, identical under both arms |
| input projection | 1.6–2.4 | wider under the bond set (two blocks, not one) |
| bond pool + hetero pool + `CutBondEncoder` (arm B) | 27.14 | **2.9× the cc pooling it replaces** |

Total per-node cost: arm A 17.20 ms vs arm B 35.89 ms — the proposal is **+109%**, not cheaper.
Narrowing the bond encoder (the `fluxmpnn_edge_proj` trick) helps but does not rescue it: at
`proj_dim=128` arm B is still 1.35× cc pooling, and only at `proj_dim=64` — a 6× lossy bond encoding —
does it reach **0.92×**, i.e. an 8% saving on a component that is 54% of per-node cost, so ~4% overall.

**Consequence.** §1's framing — "any real D4/D5 extension needs cheaper nodes" — is right, but a
fragment-representation change is not a way to get them, because per-node cost is dominated by a
term the representation does not touch. The cost argument that this proposal "actually rests on" does
not survive contact with the profiler. What remains is the *representation-quality* argument, which
is what gate §6.4 tests, and §4's complementary-pair supervision, which never depended on cost.

Boundary-only remains less lossy than it sounds: with a depth-2 GNN each boundary endpoint embedding
already summarises 2 bonds *into* the fragment interior, and the formula carries global composition.
That is now the only live reason to want it.

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
([FluxMPNN plan](fluxmpnn_plan.md#stage-3-trained-verdict-and-ceilings-from-local-memory-verified-2026-10-06)). Pair membership is recoverable
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
`docs/plans/fragnnet/sibling_ranking_target_plan.md`: a pair contributes in proportion to how exclusively its
winner's peaks are attributable to it, rather than asserting a winner outright. The 1,326 pairs/spectrum
is **not** 1,326 independent facts; the effective count is much lower and is not yet quantified.

Note also that a winner is inferred partly from the *absence* of a peak for the partner, and absence
has causes other than charge retention — below detection, aliased onto another node's m/z, or the
fragment not forming at all. This is a second reason the target must carry uncertainty rather than
commit.

## 5. What this changes elsewhere

- ~~If boundary-only holds up, node representation shrinks enough that D4/D5 depth extension — stopped
  on node cost — becomes affordable.~~ **Refuted 2026-07-23 (§2.2).** Per-node cost is dominated by
  `NodeMLP`, which is node-count-bound and indifferent to the readout; the bond-set readout measures
  +109% at faithful width and ~−4% at best under aggressive narrowing. Depth extension still needs
  cheaper nodes, but it needs *fewer* of them, not differently-described ones — that is the fan-out
  cap / breakability line, not this one.
- This is the reverted `BoundaryBondEncoder` idea (removed 2026-07-13) but as a **replacement** rather
  than an addition, with the cost argument that was not on the table then, plus directed pairs and the
  charge-site term. The earlier priority call does not bind.
- It does not address the measured bottleneck (generalization, not capacity —
  [capacity finding](../../findings/feature_headroom_2026_09_22.md#overfit-diagnostic-a6s5it2m-vs-baseline-zhjgd01n-from-local-memory-verified-2026-10-06)); the cost argument is what carries it.

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

3. **Oracle reference for radius.** — **MEASURED, 2026-07-22.** ICC of *observed* branching
   (`log q_obs + log k`, excess over uniform) by bond environment. 200 mols / 4,360 spectra /
   296,042 alias-free sibling groups / 173,930 edges.

   | radius | ICC (all edges) | noise-corrected | buckets | ICC (fixed edge set) | corrected |
   |---|---|---|---|---|---|
   | 0 | 0.0157 | 0.0214 | 40 | 0.0164 | 0.0223 |
   | 1 | 0.0936 | 0.1277 | 400 | 0.0945 | 0.1284 |
   | 2 | **0.1331** | **0.1817** | 696 | 0.1331 | 0.1808 |

   Replicate noise variance 1.6403 per observation, mean 7.64 observations/edge, noise contribution to
   edge means 0.5454. The fixed-edge-set columns hold the edge population constant across radii so the
   r0→r2 trend is not a sample-composition artifact.

   Three findings:

   - **r=2 is justified.** Observed branching does not saturate at r1; r2 adds +42% relative. This
     reverses the §2.1 counter-evidence.
   - **The trained β was a bond-type lookup table.** Bond environment explains ~72% of learned β but
     only ~18% of observed branching — a 4× gap in the wrong direction. The flow model did not fail to
     learn a function of local bond environment; it learned one *too well*, collapsing branching onto
     something far more deterministic than reality. This refines the [FluxMPNN Stage 3 verdict](fluxmpnn_plan.md#stage-3-trained-verdict-and-ceilings-from-local-memory-verified-2026-10-06)
     post-mortem: β was over-determined by exactly the descriptor later used to probe it, which is also
     why that probe looked clean.
   - **~18% is a ceiling on any bond-environment-only scorer.** Relevant to the cheap non-GNN
     BFS-time scorer proposed in [bond breakability mask plan](bond_breakability_mask_plan.md): plausibly enough for *pruning*
     which bonds to enumerate, not enough for *ranking* flux. Also bears on §2.2 — a boundary-only
     representation inherits this limit unless the heteroatom pool and formula carry the rest.

   *Caveat:* the correction treats all within-edge variance as noise, but at 7.64 observations/edge that
   variance is mostly real CE dependence, not measurement error. So 0.1817 is the right number for
   "how much of the *CE-averaged* edge signal is bond-environment-determined," and the large residual is
   consistent with the branching headroom probe ([FluxMPNN plan](fluxmpnn_plan.md)) — branching is strongly CE-dependent, and no CE-agnostic
   descriptor reaches it. That is an argument for the per-spectrum soft targets in
   `docs/plans/fragnnet/sibling_ranking_target_plan.md` rather than another architectural module.
4. **Representation probe.** — **MEASURED, 2026-07-23.** Predict observed sibling ordering from
   competing readouts over an identical 2-layer molecule GNN trunk, trained at equal budget, scored
   on **held-out molecules**. Cache: 882 train mols / 105,437 groups, 346 val mols / 40,707 groups
   (alias-free rate 26.3% and mean explained-intensity 0.261, both matching the §2 audit's 25.3% /
   0.285, so the target builder reproduces). Every arm also receives the same composition + CE
   features; arms differ **only** in the structural readout. 3 seeds.

   | arm | params | KL | TV | dom_mismatch |
   |---|---|---|---|---|
   | `comp` (composition + CE only) | 18,433 | 0.5996 ±0.0076 | 0.4501 ±0.0022 | 0.5696 ±0.0145 |
   | `het_only` (heteroatom pool) | 179,201 | 0.6165 ±0.0101 | 0.4074 ±0.0032 | 0.5046 ±0.0167 |
   | `cc` (today's readout) | 179,201 | 0.5601 ±0.0109 | 0.3777 ±0.0030 | 0.4553 ±0.0039 |
   | `cc` capacity control | 275,201 | 0.5675 ±0.0245 | 0.3776 ±0.0042 | 0.4573 ±0.0013 |
   | `bond_only` (cut bonds) | 244,993 | 0.5600 ±0.0062 | 0.3634 ±0.0023 | 0.4432 ±0.0056 |
   | `cc_plus_bond` | 261,377 | 0.5715 ±0.0276 | 0.3624 ±0.0050 | 0.4394 ±0.0109 |
   | **`cc_plus_pair`** (**production today**) | 228,481 | 0.5642 ±0.0101 | 0.3611 ±0.0032 | 0.4431 ±0.0093 |
   | **`bondset`** (the proposal) | 261,377 | 0.5570 ±0.0070 | **0.3580 ±0.0015** | **0.4364 ±0.0072** |
   | `bondset` capacity control | 174,929 | 0.5623 ±0.0074 | 0.3607 ±0.0010 | 0.4378 ±0.0037 |
   | shuffled-target control | 179,201 | 0.6261 ±0.0005 | 0.4702 ±0.0002 | 0.6924 ±0.0132 |
   | uniform floor | — | 0.6259 | 0.4700 | 0.6489 |

   **The controls hold.** The shuffled arm lands exactly on the uniform floor (KL 0.6261 vs 0.6259,
   TV 0.4702 vs 0.4700) — no signal leaks through group structure or gradient magnitude. And `cc`
   does **not** improve with 54% more parameters (TV 0.3777 → 0.3776, dom 0.4553 → 0.4573), so no
   arm's advantage here is a capacity artifact. That independently reproduces
   [capacity finding](../../findings/feature_headroom_2026_09_22.md#overfit-diagnostic-a6s5it2m-vs-baseline-zhjgd01n-from-local-memory-verified-2026-10-06) on a completely different task.

   **Cut-bond information beats the atom mask — but production already has it.** Against a *cc-only*
   baseline the proposal looks strong (TV 0.3777 → 0.3580, dom 0.4553 → 0.4364, many σ). That is the
   wrong baseline: production runs `cc` **and** `boundary_pair` together. Against `cc_plus_pair`
   (TV 0.3611, dom 0.4431) the proposal gains **0.0031 TV (~2σ) and 0.0067 dom_mismatch (~1σ)** —
   at or below this probe's resolution.

   **Where the win actually lives.** `bond_only` (0.3634 / 0.4432) already beats `cc` on its own, and
   `cc_plus_bond` (0.3624 / 0.4394) is no better than `bond_only` — so once cut bonds are present the
   atom mask contributes ~nothing, exactly as §2.2 hoped. But `boundary_pair` banks nearly all of that
   already, despite being the documented-degenerate linear encoder. The `CutBondEncoder` fix is real
   but small on this task. The heteroatom pool is a minor additive term (`bond_only` 0.3634 →
   `bondset` 0.3580) and is weak alone (`het_only` 0.4074), so §3 is not carrying the proposal either.

   **CORRECTION 2026-07-23 — the above was measured under the wrong objective.** Every arm in the
   table receives composition side-features (element counts, heavy-atom count, exact mass). Under the
   *regularization* objective (§7) that channel is itself the problem, and its presence masked the
   result: re-run without it, the ranking of arms changes completely and the conclusion "§3 is not
   carrying the proposal" is **wrong**. See §7.

   *Caveats:* this scores sibling ranking only — 26% of parent-spectra and 0.26 of intensity — not
   spectrum cosine; the probe is a 2-layer GNN on 882 molecules, not the production model; and
   `dom_mismatch ≈ 0.44` here sits in the same band as production's measured 0.45–0.50
   (the FluxMPNN Stage 0 gate ([plan](fluxmpnn_plan.md))), which is itself notable — a tiny probe matches the
   production model's branching quality.

All four are offline and need no training run. 1–3 are done; **4 is the one that decides the
proposal** and is not started.

Probe scripts for gates 1–3 currently live in the session scratchpad
(`check_complementary_pairs.py`, `check_pair_charge_headroom.py`, `check_obs_branch_icc.py`) and have
not been promoted to `scripts/eval/`. The third is the one worth keeping — it is the only instrument
in the repo that measures bond-environment explanatory power against truth rather than against a
model.

---

## 7. The regularization reframe (2026-07-23)

The motivation recorded in §1–§2.2 — cheaper nodes — was the wrong objective. The actual goal is:

1. **focus on the cut**, not the fragment;
2. **make the model less overfit by giving it a less identifiable fragment feature.**

Point 2 makes this a *regularization* proposal, which lines it up with the largest unclaimed number
in the project ([capacity finding](../../findings/feature_headroom_2026_09_22.md#overfit-diagnostic-a6s5it2m-vs-baseline-zhjgd01n-from-local-memory-verified-2026-10-06): capacity is not the bottleneck, regularization
is worth ~+0.18 val and is unswept), and with the [cc-corruption probe](../../findings/fragnnet/cc_corruption_probe_finding.md) (cos 0.82 → 0.61 under
mask corruption, magnitude unverified = the model **does** lean on exact fragment identity). Under this objective the
right metric is the train→val gap, not fit quality — which §6.4 never measured.

### 7.1 How identifying is each descriptor

882 train mols / 57,925 fragments. Distinct fragments sharing one descriptor bucket:

| descriptor | buckets | frags/bucket | median | p99 | singleton % |
|---|---|---|---|---|---|
| cc atom mask (today) | 57,925 | 1.00 | 1 | 1 | 100.0% |
| composition only | 10,480 | 5.53 | 2 | 47 | 32.5% |
| cut-bond multiset | 2,945 | **19.67** | 3 | 342 | 29.1% |
| cut-bond + composition | 38,460 | 1.51 | 1 | 6 | **72.6%** |

The mechanism is real — a cut-bond descriptor collides ~20 fragments per bucket where the atom mask
collides none. **But adding composition re-identifies the fragment** (29% → 72.6% singletons). The §2
formulation includes "+ formula / composition features"; under this objective that term actively
defeats the design.

*Caveat:* the cc row is partly tautological (keyed by the atom set itself), and the learned readout
mean-pools embeddings, which is not injective. cc is less perfectly identifying than shown — but far
closer to injective than a bond multiset.

### 7.2 Generalization gap by representation

Same probe as §6.4, now reporting train and val on the same metric. 5 seeds for the two headline arms.

| arm | comp? | trainTV | valTV | **gapTV** | gapKL | **dom_mismatch** |
|---|---|---|---|---|---|---|
| cc | yes | 0.2715 | 0.3773 | 0.1058 | 0.2349 | 0.4567 |
| bond_only | yes | 0.2123 | 0.3625 | 0.1502 | 0.3281 | 0.4418 |
| bondset | yes | 0.1974 | 0.3586 | 0.1612 | 0.3471 | 0.4375 |
| het_only | yes | — | 0.4074 | — | — | 0.5046 |
| cc | **no** | 0.2777 | 0.3850 | 0.1073 | 0.2453 | 0.4732 ±0.0094 |
| bond_only | **no** | 0.2258 | 0.3776 | 0.1518 | 0.3380 | 0.4620 |
| bondset | **no** | 0.2071 | 0.3592 | 0.1521 | 0.3416 | 0.4363 |
| **het_only** | **no** | 0.3294 | 0.4247 | **0.0953** | **0.2304** | **0.4046 ±0.0097** |

**Static cut-bond encoding (no GNN) — the largest effect on the overfitting axis.** The arms above all
feed the cut bond *k=2 GNN embeddings*, so each endpoint carries its whole radius-2 neighbourhood —
a fingerprint of the cut site, not a bond type. Replacing that with the **raw static** atom + bond
features of the cut bond (no message passing at all):

| arm (no comp) | trainTV | valTV | **gapTV** | **gapKL** | **valKL** | dom_mis |
|---|---|---|---|---|---|---|
| **static_bond** (no GNN) | 0.3731 | 0.4040 | **0.0309** | **0.0690** | **0.5482** | 0.4893 ±0.0038 |
| static_bond + het | 0.2623 | 0.3727 | 0.1104 | 0.2593 | 0.5732 | 0.4532 ±0.0051 |
| cc | 0.2789 | 0.3863 | 0.1073 | 0.2452 | 0.5761 | 0.4734 ±0.0092 |
| het_only (k=2) | 0.3299 | 0.4234 | 0.0935 | 0.2299 | 0.6404 | 0.4020 ±0.0090 |

- **3.5× smaller generalization gap** than cc (0.0309 vs 0.1073 TV; 0.0690 vs 0.2452 KL) **and the best
  held-out likelihood of any arm run** (valKL 0.5482). The least expressive representation generalizes
  best — the regularization thesis in its strongest form.
- It gets there by barely fitting at all (trainTV 0.373 vs valTV 0.404): genuinely tied, no
  memorization available. Underfit, not saturated.
- **But it is worst at ranking** (dom_mis 0.4893). Static bond features know which bonds break; they
  know nothing about which side keeps the charge.
- Combining with the heteroatom pool re-inflates the gap to 0.1104, because that term still uses k=2
  embeddings — the memorizing channel returns through the side door. **A charge-site term must be
  static/lossy too.**

**The metrics dissociate completely.** `static_bond` wins val KL and loses dom_mis; `het_only` wins
dom_mis and loses val KL. They solve different halves — cleavage vs charge retention. Any design has
to carry both, in tied form.

Three further results, none of which §6.4 could see:

- **The cut-bond arms overfit MORE, not less.** `bondset` has the largest train→val gap of any arm
  (0.1612 TV vs cc's 0.1058) — it fits training far harder for a small val gain. The expressive
  `CutBondEncoder` works *against* the regularization goal. Point 1 as implemented does not deliver
  point 2.
- **The least identifiable arm wins the ranking metric.** `het_only` with no composition has the best
  dominant-sibling accuracy of any arm run (0.4046 vs cc 0.4732, **~7σ over 5 seeds**) and the
  smallest generalization gap, *despite the worst training fit*. That is the regularization mechanism
  behaving exactly as predicted.
- **Removing composition is what unlocks it** — `het_only` 0.5046 → 0.4046, the largest single effect
  measured. It *hurts* cc (0.4567 → 0.4732) and `bond_only` (0.4418 → 0.4620), so this is not a
  generic "fewer features is better" effect; it is specific to the de-identified representation.

**Dissociation to carry:** `het_only` has the *worst* TV and val KL while having the *best*
dominant-sibling accuracy. It is underfit on distribution matching but correctly ordered — plausibly
"flat but right", which suits the sibling-ranking stop-test (`dom_mis`) but may not suit spectrum
cosine. Do not assume this transfers to cos without testing.

Chemically this is the §3 charge-site story (ionization site governs which side keeps the charge),
not the §2 cleavage story — and it says the charge-site term wants to be *lossy*.

### 7.3 What to do

`cc_atom_dropout` already exists as a training knob (`fragnnet_model.py:246`, `config/template.yml:559`)
and its docstring is literally this hypothesis: *"Regularizes against memorizing exact atom sets
(targets the train→val gap)."* A sweep is already written —
`config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_cc_atom_dropout_p{00,01,02}.yml` — and has
no recorded results, i.e. built but never run.

1. **Run that sweep.** Zero new code; it is the production form of §7.2, now with offline evidence.
   Caveat: it tests *uniform* de-identification, while §7.2 favours a structured version.
2. **Then try structured de-identification.** `cc_atom_dropout` drops atoms uniformly; §7.2 says the
   heteroatoms are the part worth keeping. Dropping carbons preferentially — keep-heteroatoms dropout —
   is a few lines on the existing knob and is what the probe actually favours.

### 7.3b Bond-encoder depth: k=1 is the optimum

The k=2 arms above were not just "a GNN" — they gave the bond encoder radius-2 endpoint embeddings.
Sweeping the depth of the trunk feeding the cut-bond encoder (no comp, 3 seeds):

| bond encoder | gapTV | valTV | valKL | **dom_mis** |
|---|---|---|---|---|
| k=0 (static, no GNN) | **0.0309** | 0.4040 | **0.5482** | 0.4893 ±0.0038 |
| **k=1** | 0.1271 | **0.3724** | 0.5778 | **0.4501 ±0.0044** |
| k=2 | 0.1504 | 0.3764 | 0.5886 | 0.4640 ±0.0086 |

**k=1 dominates k=2 on every axis** (smaller gap, better val TV, better ranking) and beats k=0 on TV
and ranking. dom_mis is non-monotonic in depth — 0.489 → **0.450** → 0.464 — so k=1 is a true
optimum, not a compromise. Same ordering for the combined arm (`bondset` k1 gap 0.1406 / dom 0.4330
vs k2 0.1524 / 0.4368).

The **charge term tolerates more depth than the bond term**: `het_only` k1 has the smallest gap of any
GNN arm (0.0721 vs 0.0934) but slightly worse ranking (0.4158 vs 0.4045). Chemically consistent —
ionization site is a property of an atom's environment; cleavage is a property of a bond.

**Reconciling with gate §6.3, which found r2 > r1 (+42%).** Not a contradiction. Gate 3 measured a
*discrete tied bucket* in a population variance decomposition, where overfitting cannot occur. The k=2
arm here is the *untied learned* version, which has r2's resolution **and** the capacity to memorize;
at this data scale the variance cost exceeds the bias gain. So the r2 signal is real but only reachable
in **tied** form — which sharpens §7.4 rather than undermining it.

Current best-known ordering for the cut term:
**static r2 bucket (untested, predicted best) > k=1 GNN (best measured learned) > k=0 static (best
calibration, underfit) > k=2 GNN (memorizes).**

### 7.4 The design §7 actually implies

Not the §2 formulation. What the measurements point at:

```
cut     -> STATIC radius-2 bond environment  (tied, ~20 frags/bucket, no fragment identity)
           fallback if that underperforms: k=1 GNN bond encoder (measured best learned, §7.3b)
charge  -> lossy/static heteroatom descriptor (NOT k=2 GNN embeddings)
formula -> OMIT (re-identifies the fragment: 29% -> 73% singletons, §7.1)
```

Rationale per term: static r2 is strictly more expressive than the `static_bond` arm above while
staying tied — gate §6.3 measured r1→r2 as **+42% relative** on real branching, and `static_bond` is
underfit, so the headroom is real. Gate §6.3 also *caps* it: bond environment explains only ~18% of
branching variance, so a static cut term alone will not reach `het_only`'s ranking — hence the second
term. And the third term has to go, or it undoes the first two.

This is close to CFM-ID's feature-tied break parameters, which is notable given
[FluxMPNN Stage 3 verdict](fluxmpnn_plan.md#stage-3-trained-verdict-and-ceilings-from-local-memory-verified-2026-10-06): the trained β collapsed to a bond-type lookup (ICC 0.72 on bond environment
vs 0.18 for reality). The model was already trying to be tied; this hands it a tied parameterization
instead of letting it fake one.

**Untested.** The static-r2 arm has not been built. That is the next offline experiment.
