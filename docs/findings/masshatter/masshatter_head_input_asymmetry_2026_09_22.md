# Do the proposal and intensity heads see the same inputs?

> **Status:** FINDING — source and config audit plus one W&B re-reading, 2026-09-22. No training
> launched. §1-§6 are read off live code on `research/masshatter-ion-h-selection` and a census of
> all 223 `config/massshatter/**` files resolved against `config/template.yml`; §7 re-reads an
> existing 630-epoch run pair against that wiring and reaches the practical conclusion.
>
> **No.** They share the same encoder output tensor and the same context vector, but they
> consume them at different granularity, and in the **entire current `ma_mi_merged` production
> lineage the proposal branch reads both inputs `.detach()`ed** — so proposal supervision
> (`proposal_loss_weight: 0.1`) never updates a single encoder or context parameter. It trains
> two head MLPs sitting on a representation shaped only by the intensity loss.
>
> **And that is apparently correct.** The one run that connects the gradient (§7) is −0.011 to
> −0.013 behind its same-seed control across epochs 100-400. Do not "fix" the detachment.
>
> **Open lead (§7.1c):** a no-training corruption probe now exists for the pooled candidate
> token. At toy scale the train/validation asymmetry sits entirely in the **boundary block** —
> the 384 columns `candidate_atom_dropout` deliberately leaves exact — while the cc pools
> behave like a generalizing feature. Needs a production checkpoint to mean anything.
>
> **See also:** the detachment census and the §7.1 gradient-connection result are consolidated
> with the loss-side reviews in [MASShatter proposal loss — consolidated audit](masshatter_proposal_loss_audit_2026_09_22.md).

Companion to the [feature set audit](../../reference/masshatter_feature_set_audit.md), which
documents the intensity-head candidate token in §4 but never states what the proposal heads
receive, and to the [proposal review](masshatter_proposal_review_2026_09_13.md), which covers
candidate *selection* rather than head *inputs*.

## 1. The wiring

One `precursor_encoder` GNN, one `MassShatterContextEncoder`, then two branches:

| | proposal branch | intensity branch |
|---|---|---|
| heads | `bond_break_head` (`BondFocusedBreakHead`, `bond_num_layers: 3` in 209/223 configs), `charge_head` (`AtomChargeAffinityHead`), optional `proposal_mixture_head` / `mechanism_head` / `ion_mode_proposal_head` | `intensity_head` (`CandidateIntensityTransformerHead`) |
| atom input | `proposal_atom_embeddings` | `atom_embeddings` |
| context input | `proposal_context` | `context` |

With `separate_proposal_encoder: False` (the whole `ma_mi_merged` lineage),
`proposal_atom_embeddings` **is** `atom_embeddings` and `proposal_context` **is** `context`,
optionally detached
([`mass_shatter_model.py:2506-2512`](../../../src/fragnnet/model/mass_shatter_model.py#L2506-L2512),
[`:2534`](../../../src/fragnnet/model/mass_shatter_model.py#L2534)). The same tensors, same values,
same dropout draw.

## 2. What each head actually receives

Production shape `mol_hidden_size: 384`, `mass_shatter_context_dim: 128`.

| input | proposal (`bond_break_head` / `charge_head`) | `intensity_head` |
|---|---|---|
| atom embeddings | same encoder, **detached** in the production lineage | live, gradient flows |
| context vector | same vector, detached; **CE stripped entirely** under `share_proposal_across_ev` | full, plus optional raw `ce_features` for FiLM / pair bias |
| granularity | per **bond** (two endpoint embeddings + `chemical_bond_attr`) and per **atom** | per **fragment**, mean-pooled |
| complement embedding | — | 384 |
| boundary embedding | — | 384, through `boundary_bond_encoder` (on in 221/223) |
| cut count, fragment size | — | 2 scalars |
| mass features (`log1p` neutral mass, precursor mass fraction) | — | 2, on in 221/223 |
| Gasteiger charge sum | — | 1, on in 221/223 |
| complement composition | — | available, on in 0 production runs |
| severed bonds' own attributes | **present** (`chemical_bond_attr`, direct) | only under `candidate_cut_bond_features`, on in 1 config |
| base proposal log-probability | — | **off**: `use_base_probability_feature = not proposal_anchored`, and `proposal_anchored: True` in 185/223 configs |
| cross-candidate attention | none — bonds scored independently | induced-set transformer over the molecule's candidates |
| `candidate_atom_dropout` regularization | not applied | applied |

Token width for the intensity head is `3·384 + 128 + 2 + 2 + 1 = 1,285` in the current lead
arm (plus `2 × edge_feature_dim` in the one `cut_bond_features` arm); the bond head projects
`2·384 + edge_feature_dim + 128` into a 512-wide refinement stack.

## 3. Gradient routing — the census

`detach_proposal_inputs = proposal_selection_only and not proposal_input_gradients_enabled`
([`:2417-2419`](../../../src/fragnnet/model/mass_shatter_model.py#L2417-L2419)).
`proposal_input_gradients_enabled` initializes to `False`
([`:1364`](../../../src/fragnnet/model/mass_shatter_model.py#L1364)) and is only ever written by
`_apply_staged_training`, which **returns early when `proposal_only_until_epoch is None`**
([`mass_shatter_pl.py:1124-1137`](../../../src/fragnnet/pl_model/mass_shatter_pl.py#L1124-L1137)).

Resolving all 223 configs against the template (2026-09-22 snapshot; the tree is actively growing, so re-run before quoting):

| steady-state routing | configs | which |
|---|---:|---|
| **proposal inputs DETACHED for the whole run** | **139** | all of `ma_mi_merged` except 5, i.e. every current arm |
| shared encoder, gradients always on (staging set, so `proposal_encoder is None` forces `True`) | 38 | `merged/warmup_*`, `subformula_filter`, `context_ablation` |
| `proposal_selection_only` off, no detach | 46 | `legacy/`, `merged/baseline`, unmerged baselines |

The 38-config middle row is a quirk worth naming: when staging is configured **and** the encoder
is shared, `set_proposal_input_gradients_enabled(proposal_only or proposal_encoder is None)`
evaluates to `True` in *every* epoch, so those runs never enter the detached regime the flag
name implies. Detachment and staging are therefore anti-correlated by construction.

**Consequence for the current lineage.** `proposal_loss_weight: 0.1` is active, but its gradient
terminates at the `bond_break_head` and `charge_head` MLP weights. The GNN that produces the atom
embeddings those heads score, and the context encoder that conditions them, are trained by the
intensity objective alone. The proposal head is asked to rank bond cleavages using a
representation selected for allocating intensity among already-chosen fragments.

## 4. Seven asymmetries, ranked

1. **Proposal supervision trains no representation** (§3). Deliberate per the code comment
   ("V1/V2 detach proposal inputs from the intensity encoder"), but the alternative it names —
   `separate_proposal_encoder`, V3 — is off in all 83 `ma_mi_merged` configs, so in practice
   neither branch owns a proposal-shaped representation.

2. **The two branches are fully decoupled at the output too.** With
   `proposal_selection_only: True`, `corrected_fragment_logits = fragment_residuals`, i.e. the
   proposal probability is discarded from the final spectrum; with `proposal_anchored: True`
   it is also absent from the candidate token. The proposal chooses *which* fragments exist;
   the intensity head then scores them with no knowledge of how confident the proposal was.

3. **`IonModeProposalHead` is the only proposal head with fragment-level inputs, and it is a
   stripped copy** — `3·384 + 128 + 2` ([`bond_fragment_heads.py:466-471`](../../../src/fragnnet/model/bond_fragment_heads.py#L466-L471)).
   No mass, charge, composition or cut-bond features, no cross-candidate attention. Off in all
   6 configs that set `separate_ion_mode_proposal_head`.

4. **`candidate_atom_dropout` does not reach the proposal pool.** The intensity path pools
   `pooled_atom_indices` ([`:2804-2815`](../../../src/fragnnet/model/mass_shatter_model.py#L2804-L2815));
   the ion-mode proposal head pools raw `fragment_atom_indices`
   ([`:2851-2860`](../../../src/fragnnet/model/mass_shatter_model.py#L2851-L2860)). The
   `_candidate_atom_dropout` docstring claims the intervention is "confined to the pooled
   component embeddings" — proposal-side pooled component embeddings escape it. Latent while
   that head is off; a silent inconsistency the moment both are on.

5. **The proposal boundary pool keeps the degenerate construction.** Intensity boundary
   embeddings route through `boundary_bond_encoder`, which exists precisely because
   endpoint-averaged bond embeddings collapse under the subsequent mean pool
   ([feature set audit §7.4](../../reference/masshatter_feature_set_audit.md)). The proposal-side
   pool ([`:2865-2875`](../../../src/fragnnet/model/mass_shatter_model.py#L2865-L2875)) uses the
   plain endpoint mean.

### 4.6 There is no separate proposal *context* encoder

`separate_proposal_encoder` gives the proposal branch its own GNN, but nothing gives it its own
`MassShatterContextEncoder`. Even under V3, `proposal_context = context.detach()`
([`:2610`](../../../src/fragnnet/model/mass_shatter_model.py#L2610)) — the detach stops the
gradient, it does not give the branch its own conditioning. The vector every proposal head reads
is therefore shaped 100% by the intensity objective, in every configuration this repository has.

### 4.7 In the shared-eV path the ion-mode head gets CE back

`share_proposal_across_ev` exists to make the proposal CE-independent, and builds
`shared_context` with `include_collision_energy=False`. The bond, charge, mixture and mechanism
heads read that CE-free vector. The ion-mode proposal head instead reads `proposal_context`,
built from the **full CE-conditioned `context`**
([`:2551-2556`](../../../src/fragnnet/model/mass_shatter_model.py#L2551-L2556), consumed at
[`:2964`](../../../src/fragnnet/model/mass_shatter_model.py#L2964)). One proposal head is
CE-conditioned while the rest are not, which contradicts the sharing invariant. Dormant: the head
is off in 223/223 and sharing is on in 2.

## 5. Checked and clean

- `share_proposal_across_ev` feeds `mol_pyg.x` rather than `atom_inputs` to `precursor_encoder`
  ([`:2400`](../../../src/fragnnet/model/mass_shatter_model.py#L2400)), which would silently drop
  context. The constructor rejects `context_in_gnn + share_proposal_across_ev`
  ([`:1367`](../../../src/fragnnet/model/mass_shatter_model.py#L1367)), so it cannot fire.
- The shared-layout path validates that grouped spectra carry identical atom features, bond
  features, bond topology, adduct, instrument and fragmentation mode before reusing one
  proposal forward.
- `separate_proposal_encoder` is rejected without `proposal_selection_only`
  ([`:1218`](../../../src/fragnnet/model/mass_shatter_model.py#L1218)), so V3 cannot be half-enabled.

## 6. Limits

This is a static audit. It establishes **what the heads are wired to read**, not that any of it
costs held-out quality. Nothing here is a measurement: no run was launched, no ablation exists
for the detached-proposal regime against a gradient-connected one on the current lineage, and
the [overfit read-out](masshatter_overfit_readout_2026_09_22.md) separately shows ~98% of the
train/validation cosine gap is intensity allocation rather than coverage — which bounds how
much a proposal-side fix could be worth on the headline metric.

## 7. Can the setup be improved? (2026-09-22)

### 7.1 The obvious fix — connect the proposal gradient — already ran, and lost

The natural conclusion from §3 is "let the proposal loss train the encoder". **That experiment
exists and nobody knew it did.** `proposal_only_until_epoch` does *two* things when the encoder
is shared, and only one of them is in its name:

```python
self.model.set_proposal_input_gradients_enabled(
    proposal_only or self.model.proposal_encoder is None   # <- always True when shared
)
```

With `separate_proposal_encoder: False`, `proposal_encoder is None`, so the flag is `True` in
**every** epoch, not just the warmup ones. Setting the key to any integer therefore un-detaches
the proposal branch for the whole run.

`jvaj1lyw` (`..._lb64_warmup_e0_4`) vs `tkkesus9` (`..._lb64_long_control`): **same seed
397623, one differing config key** (`proposal_only_until_epoch` 4 vs `None`), both 630 epochs,
both `separate_proposal_encoder: False`. Verified live against W&B, `val_cos_sim_0.01_epoch/mean`:

| epoch | control (detached) | gradient-connected | delta |
|---:|---:|---:|---:|
| 32 | 0.7132 | 0.7049 | −0.0083 |
| 64 | 0.7232 | 0.7146 | −0.0086 |
| 100 | 0.7292 | 0.7179 | **−0.0113** |
| 200 | 0.7271 | 0.7190 | −0.0080 |
| 300 | 0.7287 | 0.7157 | **−0.0130** |
| 400 | 0.7302 | 0.7170 | **−0.0132** |
| 500 | 0.7271 | 0.7209 | −0.0062 |
| 629 | 0.7309 | 0.7288 | −0.0020 |
| best | 0.7318 @555 | 0.7314 @623 | −0.0004 |

**The confound does not survive the shape.** The single key bundles a 4-epoch intensity-head
freeze with the permanent gradient connection, so a deficit at epoch 32 is uninformative. But
validation cosine on this lineage **saturates by ~epoch 100** ([horizon probe
`tkkesus9`](../../project-state.md) — 0.7292@100 against 0.7309@629, the last 529 epochs buying
+0.0017). A four-epoch initialization handicap cannot still be worth −0.0113
at epoch 100 and −0.0132 at epoch 400 in a run that stopped improving at 100. The persistent
component is the gradient connection, not the freeze.

**Conclusion: the detached proposal branch is not a bug to fix.** Coupling proposal supervision
into the shared representation is worth roughly −0.01 across every horizon a screen would read,
8× the 0.0016 replicate floor. n=1 and confounded, so this is evidence, not proof — but the
direction is the opposite of the intuition, and the cost of re-testing is a 9-hour run.

**Corollary — 38 historical runs are confounded.** Every config with staging *and* a shared
encoder (`merged/warmup_*`, `merged/subformula_filter`, `merged/context_ablation`, §3 middle
row) was silently also running gradient-connected. They are not clean tests of their nominal
variable, and the −0.01 measured here is inside their error bars.

### 7.1b The pooled-embedding memorization hypothesis, and why the prepared arm under-tests it

Raised 2026-09-22: is the **per-fragment mean pool** the memorization surface? The wiring says
it is the right place to look. Pooled atom embeddings are `3 x 384 = 1,152` of the candidate
token's **1,285 columns (89.6%)**, they are the only per-candidate input that scales with
molecular identity, and `e2_cc_atom_drop01`'s own header measures the cc atom mask at **1.00
distinct fragments per descriptor bucket — 100% singletons** on 882 molecules / 57,925 fragments.
`_candidate_atom_dropout`'s docstring names exactly this threat ("memorizing exact atom sets"),
and the [overfit read-out](masshatter_overfit_readout_2026_09_22.md) puts ~98% of the
train/validation gap in the head this token feeds. **It is untested: W&B returns 0 runs with
`candidate_atom_dropout > 0` and 0 with `candidate_cut_bond_features`.**

**But the prepared arms cannot cleanly refute it.** `candidate_atom_dropout` perturbs the
fragment and complement pools (768 columns) and deliberately leaves the **boundary embedding
exact** — the docstring says so, and the call site passes raw `fragment_atom_indices` rather than
the dropped `pooled_atom_indices`
([`:2820-2831`](../../../src/fragnnet/model/mass_shatter_model.py#L2820-L2831)). That is another
**384 identity-bearing columns**, and with `boundary_bond_encoder: True` (221/223) it encodes
oriented (retained, lost) endpoint pairs for the severed bonds — a set that, given the precursor,
largely determines the fragment. Mean pooling over cut bonds blunts this (the §7.4 degeneracy in
the [feature set audit](../../reference/masshatter_feature_set_audit.md)), so the channel is
weakened, not closed.

**Consequence:** a null from `e2`/`e3` would be ambiguous — consistent with "pooling is not the
memorization route" *and* with "identity leaked through the untreated third of the pooled
surface". Before spending the GPU, the discriminating measurement is free: take a trained
checkpoint and evaluate **train and validation cosine under corruption of the pooled blocks**
(zero or within-batch shuffle, fragment/complement alone and then including boundary). If the
train advantage collapses toward validation, the pool carries memorized signal; if both degrade
alike, it does not. No training required, and it also sizes how much dropout would have to reach
the boundary block.

### 7.1c The probe exists, and its toy-scale validation already points at the boundary block

`scripts/debug/debug_pooled_embedding_memorization.py`. It evaluates one fixed checkpoint on
a train slice and a validation slice while corrupting the pooled blocks on their way into the
intensity head, patching `CandidateIntensityTransformerHead.forward` so nothing upstream
(candidate generation, proposal scoring, the scalar features) moves. Three scopes — `cc`
(fragment + complement, 768 columns), `boundary` (384), `all` — crossed with three
substitutions: `molmean` (the molecule's mean candidate), `shuffle` (a within-molecule
permutation), `zero`. Every condition runs on the **same batches inside the batch loop**, so
the deltas are paired, and `compute_train_auxiliary_metrics` is forced on because the train
split otherwise returns no cosine at all — the gate that left train cosine unlogged in 232 of
232 runs.

**No production checkpoint is reachable from this workstation.** W&B stores none for
`tkkesus9` (0 ckpt files), the cluster checkouts hold them, and no cluster host is configured
here. Every local MASShatter checkpoint is an epoch-0 smoke run.

So the probe was validated on a **128-molecule toy model** (`debug_tiny_mass_shatter_cut_features`
at `mol_hidden_size: 64`, 60 epochs, production regime: `unique_fragment_mode` and
`proposal_selection_only` on, same `inchikey_s` split key). Reference train 0.5457 /
validation 0.3500, gap 0.1957 — it does overfit. Corrupting each block:

| condition | train | val | gap | train drop | val drop | gap closed |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.5457 | 0.3500 | 0.1957 | — | — | — |
| `cc_zero` | 0.4222 | 0.2115 | 0.2107 | 0.1235 | 0.1385 | **−0.0150** |
| `cc_molmean` | 0.4037 | 0.1966 | 0.2072 | 0.1420 | 0.1534 | **−0.0115** |
| `cc_shuffle` | 0.2586 | 0.1473 | 0.1113 | 0.2871 | 0.2027 | +0.0844 |
| `boundary_zero` | 0.3558 | 0.3087 | 0.0471 | 0.1899 | 0.0413 | **+0.1487** |
| `boundary_molmean` | 0.3523 | 0.3109 | 0.0413 | 0.1934 | 0.0391 | **+0.1544** |
| `boundary_shuffle` | 0.1899 | 0.1876 | 0.0023 | 0.3558 | 0.1624 | +0.1934 |
| `all_zero` | 0.0655 | 0.0905 | −0.0249 | 0.4802 | 0.2595 | +0.2207 |
| `all_shuffle` | 0.0437 | 0.0552 | −0.0115 | 0.5020 | 0.2948 | +0.2072 |

**At this scale the memorization signature is in the boundary block, not the cc pools.**
Destroying cc costs validation *slightly more* than train (`gap_closed` −0.012 to −0.015): it
behaves like a generalizing feature. Destroying the boundary block costs train **4.7×** what
it costs validation (0.1899 vs 0.0413 under `zero`) and closes 76% of the gap on its own;
`molmean` gives the same picture. That is the block `candidate_atom_dropout` deliberately
leaves exact — so on this evidence the prepared `e2`/`e3` arms are aimed at the wrong 768
columns.

**Weight this correctly.** 128 molecules, hidden width 64, 987 train / 424 validation spectra,
one seed. It establishes that the probe discriminates — the three scopes give clearly
different answers rather than all collapsing together — and it is a hypothesis about the
production model, not a measurement of it. Run it on a `c4a_base_s1` checkpoint before acting.

**One methodological correction the run forced:** `zero` is *not* an upper bound on a block's
contribution. `shuffle` damages more in every scope (`boundary_shuffle` 0.1899 vs
`boundary_zero` 0.3558 train), because substituting another candidate's real embedding is
confidently wrong information while a zero vector reads closer to no information. The
probe's docstring was corrected.

**Production result (2026-09-23).** The probe has now run on two production checkpoints
(`6h67zps9`, `3bej6fq7`; unmerged `[M+H]+` FT, not `c4a_base_s1`), and the toy-scale ranking
holds. At matched validation damage, boundary corruption leaves a 0.022 train/validation gap
against 0.059-0.086 for the cc pools, out of an eval-mode gap of 0.19. See the
[memorization route probe](masshatter_memorization_route_probe_2026_09_23.md).

### 7.2 What is actually worth doing

Ranked, with the reason each is not larger:

1. **Split the two behaviours of `proposal_only_until_epoch`** (config expressiveness, ~10
   lines). Today there is no way to request the intensity freeze without also connecting the
   gradient, or vice versa. This is what made §7.1 ambiguous and what confounds the 38 runs.
   Value is disconfounding future work, not cosine.

2. **Run `debug_pooled_embedding_memorization.py` on a production checkpoint** (§7.1c). It
   needs no training, and at toy scale it already separates the scopes: the cc pools behave
   like a generalizing feature while the boundary block carries the whole train/validation
   asymmetry. If that holds at production scale, `e2`/`e3` are aimed at the wrong columns and
   `candidate_atom_dropout` should be extended to the boundary pool before it is swept.
   **Done 2026-09-23; it holds on two production checkpoints**
   ([route probe](masshatter_memorization_route_probe_2026_09_23.md)).

3. **Fix the two dormant proposal-side defects** (§4.4, §4.5). `ion_mode_proposal_head` is
   `None` in **223/223** configs, so neither is live today — they are traps that would silently
   corrupt the first arm that enables it: `candidate_atom_dropout` would apply to one pooled
   component embedding and not the other, and the proposal boundary pool would use the
   construction `boundary_bond_encoder` exists to replace. Cheap, correctness only.

4. **Try the base-probability column** (`proposal_anchored: False`, one key). It is on in
   **0 of 85** `ma_mi_merged` configs, so the intensity head has never seen how confident the
   proposal was on this lineage, while `proposal_selection_only: True` also keeps that
   probability out of the output. **Caveat that keeps this at rank 4:** it opens a gradient path
   from the intensity loss *into* the proposal heads — the reverse of §7.1's coupling, untested
   in sign, and §7.1 is a reason to expect coupling the branches to cost rather than pay.

5. **Nothing on the feature axis.** Adding descriptor features to either head is closed by
   [feature headroom](../feature_headroom_2026_09_22.md) (0.93–1.00 held-out recoverability from
   current inputs), and the proposal branch's structural limit — independent per-bond Bernoullis
   with no cross-fragment competition, `proposal_num_mixtures: 1` in 223/223 — is the same line
   the connected-fragment beam and FluxMPNN work already closed.

The honest summary: **the audit itself found no performance lever** — one measurement that had
been read as something else, one config-semantics defect that confounds 38 runs, and two
dormant inconsistencies. The live candidate is §7.1c's boundary-block result, which is a
toy-scale hypothesis awaiting a production checkpoint, not a finding. That is consistent with the
[overfit read-out](masshatter_overfit_readout_2026_09_22.md), which puts ~98% of the
train/validation gap in intensity allocation rather than in anything the proposal branch controls.
