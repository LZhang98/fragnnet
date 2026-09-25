# Should we add more features? Measured: no, except one class we cannot compute

> **Status:** FINDING — CLOSED NULL for descriptor features &nbsp;·&nbsp; **Updated:** 2026-09-22
>
> Every standard 2D descriptor family is recoverable from the inputs both models already
> receive, at 0.93–1.00 held-out skill with only two hops of context. The one feature class with
> real headroom is externally-derived quantum knowledge (bond dissociation energies, protonation
> sites), and the repository's own reference test caps any bond-environment-only signal at
> **~18% of observed branching**. Read this before adding an atom or bond feature.
>
> **Second pass, shortcut framing.** Derivability is the wrong criterion on its own: a feature
> can be derivable and still not be the function the model fits. The charge-retention test below
> looked for a shortcut a "correct" feature would displace and **did not find one** — but it did
> find a real shortcut elsewhere, `depth` standing in for cut count, whose correct feature is
> already implemented and switched off.

Follows the 2026-09-22 [GLACIER parity work](../notes/glacier_object_detection.md), which closed
the gap to a competitor's feature list. This asks the next question — whether *anyone's* feature
list is the thing holding us back.

## The question a feature audit has to answer

"Is feature X predictive of fragmentation?" is the wrong test; most chemistry descriptors are.
Every 2D-topological descriptor is a deterministic function of the molecular graph, so the model
can in principle compute it. The decision-relevant question is whether the encoder **can already
derive it from its current inputs** — if it can, the feature adds no information and costs input
width and overfitting surface.

## Method

`scripts/debug/debug_feature_headroom_probe.py`. For each candidate descriptor, predict it from
the current atom/bond feature set (the 2026-09-22 parity set: 73-dim atom, 16-dim bond) plus 1-
and 2-hop neighbourhood means — the information a message-passing encoder holds after two rounds.
Random forest for categorical targets, gradient boosting for continuous, held-out split **grouped
by molecule**, 1,499 NIST20 molecules / 35,484 atoms / 37,479 bonds.

This is a **lower bound** on what the production encoders can derive. FraGNNet runs 4 GINE
layers; MASSter runs GINE_GSA, which adds exact per-graph global attention. Both are strictly
more expressive than a two-hop mean.

## Result: nothing in the descriptor families is new information

| atom candidate | source | metric | skill | baseline |
|---|---|---|---:|---:|
| `mmff_atom_type` (33 classes) | MMFF94, the typing CFM-ID uses | bal. acc | **0.959** | 0.030 |
| `mmff_partial_charge` | MMFF94 | R² | **0.991** | — |
| `estate_index` | Kier–Hall electrotopological state | R² | **0.994** | — |
| `crippen_logp` / `crippen_mr` | Crippen atom contributions | R² | **0.978** / **0.995** | — |
| `tpsa_contrib` | polar surface area contribution | R² | **1.000** | — |
| `labute_asa` | accessible surface contribution | R² | **0.997** | — |
| `is_benzylic` | SMARTS | bal. acc | **0.994** | 0.500 |
| `is_allylic` | SMARTS | bal. acc | **0.928** | 0.500 |
| `is_alpha_carbonyl` | SMARTS | bal. acc | **0.984** | 0.500 |
| `is_alpha_hetero` | SMARTS; the `BUCKET_LOSS_RANK` descriptor's own term | bal. acc | **0.999** | 0.500 |

| bond candidate | metric | skill | baseline |
|---|---|---:|---:|
| `is_rotatable` | bal. acc | **0.999** | 0.500 |
| `is_amide_bond` | bal. acc | **0.989** | 0.500 |
| `is_benzylic_bond` | bal. acc | **0.985** | 0.500 |
| `alpha_carbonyl_bond` | bal. acc | **0.972** | 0.500 |

These cover the descriptor families CFM-ID's feature set is built from — `IonRootMMFFAtomType`,
`NeighbourMMFFAtomType`, `GasteigerCharges`, the `FunctionalGroupFeature` family, `BreakAtomPair`
and `NeighbourOrigBondTypes` are all MMFF typing, partial charge, SMARTS group membership, or
local bond-type context. All of it is already computable from what we feed the encoder.

## Two existing measurements bound the rest of the argument

**A bond-environment-only scorer is capped at ~18% of observed branching.** The reference test
recorded in [`fragment_bond_set_representation.md`](../plans/fragnnet/fragment_bond_set_representation.md)
§6.3 (summarized in [`fluxmpnn_plan.md`](../plans/fragnnet/fluxmpnn_plan.md)) decomposed *observed*
branching over radius-0/1/2 bond environments on 200 molecules / 296,042 alias-free sibling
groups: noise-corrected ICC
0.0214 → 0.1277 → **0.1817**. Bond environment explains ~72% of the model's *learned* β but only
~18% of real branching. Every candidate above is a bond- or atom-environment descriptor, so 18%
is the ceiling on the whole class — including a better-encoded one. The residual is CE
dependence, which is a conditioning problem, not a feature problem.

**The model is not information-starved, it is generalization-limited.** The FraGNNet overfit
diagnostic run `a6s5it2m`, with regularization off, reaches train 0.8951 against a train oracle
of 0.9221 — **97% of the DAG ceiling** — while val sits at 0.5283 against a val oracle of 0.9292
(57%). A model that already fits 97% of what its DAG supports is deriving whatever it needs from
the inputs it has; derivable features cannot fix a generalization gap, and each one widens the
input. That run is recorded in W&B and in the agent memory store, **not** under `docs/`, so it is
cited here rather than restated — it is worth promoting into a finding of its own.

## Metadata: audited, and almost all of it is dead

Unlike descriptors, spectrum metadata is *not* derivable from the molecule, so an unused column
would be real information. `spec_df` carries four the models do not read. Measured on the
580,283-spectrum `ma_mi`-style subset (FT/QTOF, HCD/CID, the four modelled adducts):

| column | content | verdict |
|---|---|---|
| `col_gas` | N2 478,049 · **He 3,044 (0.5%)** · missing 99,190 (17%) | ~constant where known, and the missing block is confounded with instrument. Not worth a feature |
| `ce_type` | `single` 554,980 · `none` 25,303 | **no stepped or ramped spectra at all** |
| `ace_extra_*`, `nce_extra_*` | stepped-CE ladder values | **100% null** in the subset |
| `res` | 13 distinct codes; 4 (499,908), 6, 2, 5, 3 | live, but see below |

So the "we feed a single NCE for a stepped-CE acquisition" mismatch **does not exist** in this
corpus. The one live column, `res`, is not a model input question: the loss already applies a
per-instrument tolerance (`inst_type_loss_tol`, FT 1e-5 / QTOF 2e-5), and `res` varies *within*
instrument. Whether the tolerance should be per-spectrum rather than per-instrument is a separate
question about the loss, not about features. Not investigated here.

## Conclusion

1. **Do not add descriptor features.** MMFF typing, EState, Crippen, TPSA/ASA and the
   benzylic / allylic / α-carbonyl / α-heteroatom family are all already available to the model.
2. **Do not expect a better bond descriptor to help.** The class is capped at ~18% of observed
   branching, measured on our own data.
3. **The only class that would add information is external quantum knowledge** — per-bond
   dissociation energies from a QM-trained model (e.g. ALFABET, St. John et al. 2020, ~290k DFT
   BDEs) or per-site proton affinity. These are not new facts about the molecule; they transfer
   knowledge from a corpus we do not train on, which is why they are not covered by the
   derivability result above. **But** they are still bond-environment-class quantities, so the
   18% ceiling applies to them too. That ranks them below the CE-allocation work, and they should
   not be started on a feature argument.
4. The residual in every decomposition points the same way: **CE-dependent allocation**, per
   [`glacier_gap_ce_decomposition_2026_09_21.md`](fragnnet/glacier_gap_ce_decomposition_2026_09_21.md) and
   [`branching_headroom`](../plans/fragnnet/fluxmpnn_plan.md).

## Second pass: shortcut learning (2026-09-22)

The derivability criterion above answers "does this feature add information". It does not answer
"is this the function the model actually fits" — a feature can be perfectly derivable and still
lose to an easier, non-causal correlate. That is a sharper criterion and it deserves its own
test, because this repository has a **documented instance** of exactly that failure: FluxMPNN's
learned β collapsed onto a near-deterministic bond-type lookup, explaining ~72% of its own
variance but only ~18% of real branching. Note what the fix was not: the shortcut there *was* a
bond-environment feature, so adding more of the same class would have deepened it.

### Test: charge retention, the most mechanism-loaded choice the model makes

`scripts/debug/debug_charge_retention_shortcut.py`. For a cleavage that produced an observed
ion, which half keeps the charge? Built from NIST's own peak annotations, no model in the loop:
for each annotated product of an `[M+H]+` spectrum the complementary ion is computed exactly
from the formulas and looked up in the same spectrum, scoring **zero when absent** so outright
wins count. Restricted to precursors with exactly one nitrogen, which holds composition fixed
while letting the nitrogen's **basicity class** vary — a structural property no element count
can express. 1,524,474 splits over 6,430 molecules.

The competing hypotheses: the **shortcut** is "the bigger half keeps the charge"; the
**mechanism** is the mobile proton model, "the half holding the basic site keeps it".

| feature group | random molecules | InChIKey split | scaffold split |
|---|---:|---:|---:|
| size and mass only (the shortcut) | 0.7837 | 0.7659 | 0.7584 |
| composition of both halves | 0.8491 | 0.8387 | 0.8266 |
| composition + nitrogen basicity class | 0.8529 | 0.8364 | **0.8275** |

**Two null results, both against the missing-feature story.**

1. **The mechanistic feature adds nothing.** Basicity class is worth +0.004 AUC on a random
   split, **−0.002 on the InChIKey split** and +0.001 on scaffolds. Composition of both halves
   has already absorbed it. Per-class retention rates do vary (nitro 0.320, aliphatic amine
   0.390, amide 0.390, nitrile 0.420, aromatic N 0.537, at constant nitrogen count) — so class
   is not nothing — but it is not orthogonal to composition.
2. **The shortcut is not brittle.** If size/mass were a shortcut in the damaging sense, its
   skill would decay faster than the causal features when the structures change. It does not:
   random → scaffold costs size/mass −0.025, composition −0.023, composition+class −0.025. All
   three degrade identically.

So on this axis there is no shortcut for a correct feature to displace. That does not clear the
model — it locates the question elsewhere.

### A real shortcut, with a free fix that is switched off

`depth` standing in for **how many bonds were broken**. On the production DAG
(`nist20_ma_mi_d3_h4_isoFalse_smpFalse`, plain single-bond BFS) the two are identical, so
`depth` is an unbiased proxy and the model is rewarded for learning it. They stop being
identical the moment the DAG is built any other way: the SMARTS ring-cleavage rules break four
bonds in one BFS step, so a 4-cut xanthone node and a 1-cut node are both depth 1. The comment
at [`fragnnet_model.py:1719`](../../src/fragnnet/model/fragnnet_model.py#L1719) already says so
in as many words — "`depth` is NOT a proxy".

The correct feature, `num_cut_bonds`, is implemented, derived from indices already batched on
every forward pass ("costs no data and no DAG regeneration"), and **off in every config**,
including `fraggnn_d3_ma_mi_nist20_amolc_all_prepass_new_mol_feats.yml` — the SMARTS-prepass run
(`alvhehe2`), which is precisely the run where `depth` is a broken proxy, and which **lost** to
its non-prepass comparator. That loss is at minimum confounded by this; it is a cheap,
falsifiable prediction that re-running the prepass arm with `num_cut_bonds` on closes part of
the gap. The same applies to the `_mbfs` and `_ringmerge` DAGs.

This is the shape a genuine shortcut takes here, and it is worth noting what it is not: not a
missing chemical descriptor, but a **data-construction artifact** that makes a cheap feature
accidentally sufficient on one DAG and wrong on another.

### What would actually settle it

Feature-availability experiments cannot establish what a trained model relies on. The instrument
that can is an intervention on a checkpoint — corrupt or randomize one feature group at a time
and measure the prediction shift on train versus held-out structures. This repository already
has the pattern in the `cc`-corruption probe (fragment identity is genuinely used: cosine
0.82 → 0.61 under mask randomization). Running it across feature groups needs a production
checkpoint, which is not in the local tree.

## What this did not test

3D conformer geometry (genuinely not a function of the 2D graph; 3DMolMS uses it) was not
probed — no conformer pipeline exists here and generating one is a preprocessing change. The
probe measures *derivability*, not sample efficiency: a derivable feature can still help a
data-limited model as inductive bias. The capacity finding argues against that mattering here,
but it is an argument, not a measurement on these specific features. No model was trained.
