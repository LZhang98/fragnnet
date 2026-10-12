# MASSter Feature Set Audit

> **Status:** REFERENCE &nbsp;·&nbsp; **Updated:** 2026-10-12 &nbsp;·&nbsp; findings §7.1-§7.8
> are **FIXED for MASSter** and are now the default; see §11. §12 and §13 add the 2026-09-22
> GLACIER parity work: graph features, which both models can now enable, and the candidate
> token's complement composition.
>
> Which atom/bond/candidate/conditioning features MASSter actually consumes.
> Headline finding: **34 of 70 atom-feature columns were constant on NIST20**, 12 of them
> because `a_degree`'s one-hot is mis-encoded, and the candidate intensity head was mass-blind.
> Sections 1-10 describe the audited state; §11 records what changed.
> **Every MASSter checkpoint written before 2026-09-20 is incompatible with this code.**

Audited against `31c487db` on branch `research/masshatter-ion-h-selection`, over all 175
`model_type: masster` configs under `config/`. Column-constancy measured on a 5,000-molecule
sample (seed 0) of `data/proc/nist20/mol_df.pkl` (120,641 atoms, 254,498 bonds).
The companion FraGNNet document is [feature_set_audit.md](feature_set_audit.md); the metadata
conditioning design is audited separately in
[masster_context_design_audit.md](../findings/masster/masster_context_design_audit.md).

---

## 1. Where features enter

MASSter consumes features at five sites. Everything else is derived.

| Site | Code | Inputs |
|------|------|--------|
| Precursor GNN | `GNN` in [layers/gnn.py](../../src/fragnnet/layers/gnn.py) | `mol_pyg.x` (atom feats), `mol_pyg.edge_attr` (bond feats), optionally `context` |
| Bond break head | `BondFocusedBreakHead` / `EndpointBondBreakHead` | endpoint atom embeddings, `chemical_bond_attr`, `context` |
| Charge affinity head | `AtomChargeAffinityHead` | atom embeddings, `context` |
| Candidate intensity head | `CandidateIntensityTransformerHead` | 3 pooled embeddings + `context` + 3 scalars (+2 optional) |
| Context encoder | `MassterContextEncoder` | adduct, instrument, frag mode, NCE, ACE |

Chemistry (masses, valid channels, H-transfer bounds) is computed from separate non-learned
buffers; see §6. No mass or formula ever reaches a learned parameter.

## 2. Atom features (mol GNN input)

`node_feature_dim = 70` in 169/175 configs (the other 6 inherit the template's 60; see §8).
Constancy is measured over the NIST20 sample, so "constant" means "this dataset never exercises
the bucket", except for `a_degree`, which is structurally broken.

| Name | Cols | Enabled | Constant cols | Notes |
|------|------|---------|---------------|-------|
| `a_onehot` | 18 | yes | 6 | `H` (no explicit H in the graph), `As`/`Hg`/`Pb`/`Se`, and the unknown bucket never fire |
| `a_degree` | 14 | yes | **12** | **Defect.** Cols 0–1 are the raw `GetDegree()`/`GetTotalDegree()` integers and are live; cols 2–13 are the one-hot and are *always* `[0]*11 + [1]` — see §7.1 |
| `a_hybrid` | 6 | yes | 2 | SP3D2 and unknown never fire (SP3D: 4 atoms in 120,641) |
| `a_formal` | 6 | yes | 2 | charge −2 and unknown never fire; observed {0, +1, −1, +2} |
| `a_radical` | 6 | yes | **6** | Every sampled atom has 0 radical electrons — the feature is fully uninformative here |
| `a_ring` | 4 | yes | 0 | is-in-ring bits are redundant with `a_ring_size` (documented TODO in `feat_utils.py`), is-aromatic bits are not |
| `a_ring_size` | 5 | yes | 0 | `[in3, in4, in5, in6, in≥7]`; closes the FraGNNet audit's top gap |
| `a_num_hs` | 5 | yes | 1 | ≥4-H bucket never fires. Exactly `GetTotalDegree() − GetDegree()`, i.e. a re-encoding of information `a_degree` already carries — but as a one-hot it is newly usable by the input `Linear` |
| `a_mass` | 1 | yes | 0 | `GetMass() * 0.01` |
| `a_chiral` | 5 | yes | **5** | Every sampled atom is `CHI_UNSPECIFIED`; preprocessing strips stereochemistry |
| `a_charge` | 1 | **no** | — | Gasteiger partial charge. Implemented, used by FraGNNet's `pcdl_nist23*`/`nps_pcdl_nist23*` configs, **never enabled in any MASSter config** (see §7.3) |
| random-walk PE | `pyg_pe_embed_k` | **no** | — | 0 in all 175 configs |

**Total: 34/70 columns (49%) constant.** 12 are the `a_degree` coding defect; the remaining 22
are empty one-hot buckets, of which `a_radical` (6) and `a_chiral` (5) are entirely empty.
Effective width is 36.

## 3. Bond features

`edge_feature_dim = 9` in 169/175 configs. Bond features are consumed twice: `edge_transform`
inside GINE/GINE_GSA, and directly by the bond break head via `chemical_bond_attr`.

| Name | Cols | Enabled | Constant cols | Notes |
|------|------|---------|---------------|-------|
| `b_degree` | 5 | yes | 1 | unknown bucket never fires |
| `b_ring` | 2 | yes | 0 | |
| `b_conj` | 2 | yes | 0 | |
| `b_stereo` | 7 | **no** | — | Implemented; never enabled in any config in the repository |

## 4. Candidate-level features (intensity head token)

One token per retained unique fragment. In the production shape
(`mol_hidden_size: 384`, `masster_context_dim: 128`) the token input is **1,283 dims**:

| Component | Width | Live? |
|-----------|-------|-------|
| `fragment_embeddings` — mean-pooled atom embeddings of the component | 384 | yes |
| `complement_embeddings` — mean-pooled atom embeddings of the complement | 384 | yes |
| `boundary_embeddings` — mean over cut bonds of `(atom_emb[u] + atom_emb[v]) / 2` | 384 | yes |
| `context` — per-spectrum metadata vector, broadcast per candidate | 128 | yes |
| `cut_counts` — number of severed bonds, raw integer | 1 | yes |
| `log1p(component_sizes)` — atom count | 1 | yes |
| `log1p(mask_component_counts)` | 1 | **no — constant `log 2`** in unique-fragment mode |
| `charge_log_probabilities` (only if `use_base_probability_feature`) | 1 | **no — passed as `zeros_like`** in unique-fragment mode |
| `mask_log_probabilities` (only if `use_base_probability_feature`) | 1 | yes |

`use_base_probability_feature = not masster_proposal_anchored`, and 135/175 configs set
`proposal_anchored: True`, so both base-probability inputs are off in the current arms.

What is **not** in the token: fragment mass, channel m/z, elemental composition, hydrogen count,
RDBE, precursor m/z or mass fraction, partial charge, and the cut bonds' own feature vectors.
Extent is represented only by `log1p(atom count)` and `cut_counts`; pooling is `mean`, so the
size signal in the embeddings is normalized away.

## 5. Spectrum-level conditioning

Encoded once per spectrum by `MassterContextEncoder`, then broadcast to bonds, atoms and
candidates. Unlike FraGNNet there is no separate insert-location machinery — one vector, all heads.

| Feature | Field | Configs enabling it | Notes |
|---------|-------|---------------------|-------|
| NCE | `spec_nce` | 175/175 | `ce_insert_type: embed` in 152: a fixed `int_embedder` table (abs-sines by default) plus a learned linear layer, mean-pooled over merged CE readings. The abs-sines table aliases NCE x with 256 − x and gives neighbouring values unrelated codes; `ce_insert_type: qet` is the smooth alternative ([finding](../findings/masster/masster_qet_projection_candidate_union_2026_09_26.md#the-nce-input-is-categorical)) |
| ACE | `spec_ace` | 24/175 | dual-CE arms only |
| Precursor adduct | `spec_prec_type` | 175/175 | required; `prec_type: True` is enforced at setup |
| Instrument | `spec_inst_type` | 79 explicit True, 93 inherit template True | empty vocabulary skips the embedding entirely |
| Frag mode | `spec_frag_mode` | 77/175 | template default False; one extra row for unknown |
| Precursor m/z | `spec_prec_mz` | **0** | loaded and batched, used only by metrics — never a model input (same gap as FraGNNet) |
| SMARTS rule id | `rule_indices` | 2/175 | `SmartsMechanismHead` rule embedding; `masster_smarts_templates` defaults False |

`context_in_gnn` (context appended to atom inputs) is on in 3/175 configs; it is rejected at
construction together with `share_proposal_across_ev`.

## 6. Non-learned chemistry fields

Attached by `MolGraph.get_pyg_graph` and required by `MassterModel.forward`. These are
arithmetic inputs, not features — they never touch a parameter.

| Field | Read by | Purpose |
|-------|---------|---------|
| `atom_monoisotopic_masses` | `_fragment_masses_and_valid_channels` | fragment neutral mass → `ion_mzs` |
| `atom_hydrogen_counts` | same | H budget per fragment, CMF/CRF validity |
| `atom_atomic_numbers` | same | element bookkeeping |
| `atom_free_valence_slots` | `_transferable_hydrogen_shifts` | `h_transfer_bound: valence` (template default since 2026-09-20) |
| `atom_formal_charges` | **nothing** | required by the `forward` field check, never read (§7.7) |

## 7. Findings

### 7.1 `a_degree`'s one-hot is dead — 12 constant columns in every model

`AtomDegree.featurize` passes a *list* to `one_hot_encoding`:

```python
deg = [x.GetDegree(), x.GetTotalDegree()]
onehot = one_hot_encoding(deg, cls.allowable_set, True)   # allowable_set = ints 0..10
```

A list never equals an `int`, so `x not in allowable_set` is always true, `x` is replaced by
`None`, and the result is always the unknown bucket. Verified on real molecules: every atom of
aspirin returns `[deg, total_deg, 0,0,0,0,0,0,0,0,0,0,0,1]`, and columns 2–13 have zero variance
over the whole NIST20 sample.

Consequence: heavy-atom substitution degree — a first-order determinant of which bond cleaves —
reaches the GNN only as two unnormalized integers through a single `nn.Linear`. The observed
degree range is {1, 2, 3, 4}, so the intended one-hot would have needed 4 live buckets, not 11.
`tests/test_feat_utils.py::test_atom_degree_carbon` asserts only `feat[0]`, `feat[1]` and the
length, which is why this survived.

This is shared with FraGNNet, which uses the same featurizer.

Compatibility: the 12 columns are already allocated, so writing a real `GetDegree()` one-hot into
them keeps `node_feature_dim` and every checkpoint tensor shape. It still changes input
*semantics*, so an existing checkpoint loaded against a fixed featurizer silently sees different
inputs. The first fix therefore shipped a separately named `a_degree_onehot`; on 2026-09-20 the
user authorized the in-place repair under [engineering rules §6](../rules/engineering.md), and
`a_degree` itself was fixed and the duplicate removed. **Every checkpoint trained before
2026-09-20 must be retrained** — it will load without error and read different inputs.

### 7.2 The candidate intensity head is mass-blind

No m/z, fragment mass, or composition enters any learned parameter (§4). Masses exist in scope:
`ion_mzs` is computed at `masster_model.py:2264`, before the intensity head is called at
`:2402`. Feeding fragment mass, mass fraction of the precursor, or per-channel m/z into the token
is a local change to an already-computed tensor.

### 7.3 `a_charge` is never enabled for MASSter, and MASSter is the model that wants it

MASSter has an explicit `AtomChargeAffinityHead` whose whole job is deciding which component
keeps the charge, plus a CRF/CMF ion-mode split. Its atom inputs contain no electrostatic prior.
Gasteiger charges are implemented, computed on the fly (no DAG regeneration), and already used by
FraGNNet's newest configs, which also carry `frag_q` (the per-fragment Gasteiger sum). MASSter
has no `frag_q` analogue in its candidate token.

### 7.4 The boundary embedding is degenerate — FraGNNet diagnosed and fixed this construction

`boundary_embeddings` pool `(atom_emb[u] + atom_emb[v]) / 2` with `reduction="mean"`, so for a
fragment with *k* cut bonds the result is exactly
`0.5 * (mean of inside endpoints + mean of outside endpoints)`. Two consequences, both verified
numerically against `pool_boundary_bond_embeddings`:

- **Pairing is destroyed.** Cutting a1–b1 plus a2–b2 gives a bit-identical embedding to cutting
  a1–b2 plus a2–b1 (max abs difference 3e-8, float noise). That is precisely the multi-cut
  ring-cleavage regime.
- **Inside/outside is destroyed.** The per-bond term is symmetric in its endpoints, so the atoms
  the fragment *keeps* and the atoms it *loses* enter with the same weight and cannot be
  distinguished within the boundary term.

This is the same degeneracy FraGNNet diagnosed for its `boundary_pair` feature and fixed with
`CutBondEncoder` ([cut_bond_encoder.py](../../src/fragnnet/fragmentation/cut_bond_encoder.py)), whose
docstring states the argument in full: a linear map on a concatenation cannot express a bond
because bond identity is an interaction, and a linear map commutes through the pool. MASSter's
version is strictly weaker than the `boundary_pair` that fix replaced — `boundary_pair` at least
had separate `W_in`/`W_out`, so it could tell inside from outside.

Separately, `b_degree`/`b_ring`/`b_conj` for the *severed* bonds reach the intensity head only
through whatever GINE mixed into the endpoint atom states, while `BondFocusedBreakHead` (3 layers
in 156/175 configs) builds refined per-bond states that read `chemical_bond_attr` directly and
discards them after emitting break logits. Pooling those existing bond states as the boundary
source, or porting `CutBondEncoder`, is shape-local.

### 7.5 Two dead inputs inside the candidate token

`log1p(mask_component_counts)` is constant (`log 2`) in unique-fragment mode, which is 143/175
configs; the head's docstring already records this and keeps it for legacy-mode and checkpoint
compatibility. Separately, `charge_log_probabilities` is passed as `th.zeros_like(...)` on the
unique-fragment path, so when base-probability features are on, one of the two is identically zero.

### 7.6 `a_radical` and `a_chiral` carry no information on NIST20

All 120,641 sampled atoms have 0 radical electrons and `CHI_UNSPECIFIED` — preprocessing strips
stereochemistry. That is 11 of 70 columns. Removing them changes `node_feature_dim`, so it is
worth doing only bundled with another feature change that already breaks checkpoint shapes.

### 7.7 `atom_formal_charges` is required but never read

`MassterModel.forward` fails a batch whose graphs lack `atom_formal_charges`, yet no model or
chemistry code reads it (repo-wide: written in `feat_utils.py:257`, listed in the required-field
check at `masster_model.py:1904`, asserted in `tests/test_spec_mol_dataset.py`). Formal charge
is also absent from the fragment-mass arithmetic, which uses atomic numbers plus hydrogen counts.

### 7.8 The featurizer mutates the shared canonical element order

`AtomOneHot.allowable_set` *is* `formula_utils.CANONICAL_ELEMENT_ORDER` (same list object), and
`one_hot_encoding(..., encode_unknown=True)` appends `None` to it on first use. After featurizing
any molecule the shared list is 18 long instead of 17. Harmless today — its two consumers do
`if elem in elem_d`, and `ELEMENT_TO_IDX`/`IDX_TO_ELEMENT` are built from `ELEMENT_TO_VE`, not
from this list — but any length-based or lazily-built consumer would break. Shared with FraGNNet,
which is the heavier user of element-ordered formula arrays.

## 8. Config drift

| Group | Count | Feature shape |
|-------|-------|---------------|
| Experiment configs under `config/masster/**` and `config/sweeps_merged/masster/**` | 169 | 70 atom / 9 bond |
| `config/nist20_merged_inchikey/masster/s0.yml` and 5 `config/debug/debug_tiny_masster*.yml` | 6 | template 60 atom / 5 bond |

The canonical seeded config (`nist20_merged_inchikey/masster/s0.yml`) therefore does not
match the feature set every experiment arm actually trains with, and produces a different
`node_feature_dim`. `config/*_template.yml` keeps `a_ring_size`, `a_charge`, `b_ring` and `b_conj`
commented out.

Minor: `masster_pl.py:967-969` reads the three Graphormer structural keys with
`self.hparams.get(...)`, which [Python rules](../rules/python.md) forbid for hyperparameters;
all three are declared in `config/*_template.yml`, so the fallback only hides missing keys in
legacy checkpoint configs.

## 9. Suggested order of work

1. **`a_degree` one-hot** — real defect, shape-preserving fix available, needs a compatibility
   decision before touching it (§7.1).
2. **Fragment mass / channel m/z into the candidate token** — the tensor already exists (§7.2).
3. **Enable `a_charge`, add a `frag_q` analogue** — one config line plus a candidate-token scalar;
   aimed directly at the charge-retention head (§7.3).
4. **Reuse the bond head's refined states as boundary embeddings** (§7.4).
5. Cleanups when something else already breaks checkpoint shapes: drop `a_radical`/`a_chiral`,
   drop the redundant `a_ring` in-ring bits, drop the dead token columns, remove or use
   `atom_formal_charges`, align the drifted configs (§7.5–§8).

None of these are measured wins. Each needs a matched arm, read against the measured same-seed
replicate floor (0.003 val cosine) recorded in the
[regularization sweep plan](../plans/masster/masster_regularization_sweep_plan.md).

## 10. Which of these apply to FraGNNet

`SpecMolFragDataset` extends `SpecMolDataset`, so both models build their molecular graph through
the same `MolGraph`/`get_mol_graph` featurizer. Everything at the featurizer level is therefore
shared; everything at the architecture level is not. Measured on the same NIST20 sample:

| Feature set | Atom dim | Constant cols |
|-------------|----------|---------------|
| MASSter, 169/175 configs | 70 | 34 (49%) |
| FraGNNet newest (`pcdl_nist23*`, `nps_pcdl_nist23*`), 44/323 configs | 66 | 33 (50%) |
| Template, inherited by 277/323 FraGNNet configs | 60 | 33 (55%) |

| Finding | FraGNNet |
|---------|----------|
| §7.1 `a_degree` one-hot always the unknown bucket | **Yes, identical** — same 12 dead columns in all 323 configs |
| §7.6 `a_radical` (6) and `a_chiral` (5) fully constant | **Yes, identical** |
| `a_onehot` / `a_hybrid` / `a_formal` empty buckets | **Yes, identical** |
| §7.8 shared-list mutation of `CANONICAL_ELEMENT_ORDER` | **Yes, shared** — and FraGNNet is the heavier user of element-ordered formula arrays |
| `b_stereo` implemented, never enabled | **Yes, shared** — no config in the repository enables it |
| `self.hparams.get(...)` for the Graphormer keys | **Yes** — same three lines at `fragnnet_pl.py:112-114` |
| §7.4 degenerate boundary/cut-bond embedding | **Had it, fixed it.** `boundary_pair` has the same defect; `CutBondEncoder` replaced it at identical width. MASSter's symmetric mean is weaker than the version FraGNNet retired |
| §7.2 no mass/composition in the candidate token | **No.** FraGNNet fragment nodes carry `base_formula` (elemental composition), `h_formulae_idx`, `depth`, `num_heavy_atoms`, and optionally `frag_q`, `num_cut_bonds`, `rdbe`. Composition is an explicit input there and absent here |
| §7.3 `a_charge` never enabled | **Partly.** Enabled in 44/323 FraGNNet configs together with `frag_q`; the other 277 inherit the template's 8-feature set, so most FraGNNet arms also lack it. MASSter has it in 0/175 and has no `frag_q` analogue |
| §7.7 `atom_formal_charges` | **Weaker.** Attached for every model but read by none; only MASSter *requires* it in `forward` |
| §7.5 dead candidate-token columns | **No** — MASSter-specific |
| `a_ring_size`, `a_num_hs`, `b_ring`, `b_conj` | **MASSter is ahead.** All four are on in 169/175 MASSter configs; FraGNNet enables `a_ring_size`/`b_ring`/`b_conj` in 44/323 and `a_num_hs` nowhere |

So the repository-wide items are §7.1, §7.6, §7.8 and `b_stereo`; §7.4 is a solved problem in
FraGNNet that MASSter has not inherited the solution to; and §7.2 is the one place where
MASSter is missing a feature class FraGNNet has always had.

Note that the FraGNNet index row and
[feature_set_audit.md](feature_set_audit.md) (2026-06-09) still list `a_ring_size` and `b_ring` as
open gaps. They are implemented and enabled in 44 configs, so that document is stale.

## 11. Fixes applied (2026-09-20)

All of §7 is implemented for MASSter and on by default. FraGNNet keeps its previous inputs:
the shared featurizer gains a new feature rather than changing `a_degree` in place, and no
FraGNNet config was touched. **Existing MASSter checkpoints cannot be loaded**: the atom
feature width changed (70 → 71) and the intensity head's input width changed, so the affected
tensors mismatch. Nothing was trained or evaluated; these are not measured improvements.

| Finding | Fix |
|---------|-----|
| §7.1 dead `a_degree` one-hot | First shipped as a separately named `a_degree_onehot`; on 2026-09-20 repaired **in place** and the duplicate deleted. `a_degree` is now the two raw values plus a working one-hot of the heavy-atom degree, at the unchanged 14-column width. Shared with FraGNNet, so every pre-2026-09-20 checkpoint of either model must be retrained |
| §7.2 mass-blind candidate token | `masster_candidate_mass_features` (default True) adds `log1p(neutral fragment mass)` and the fragment's fraction of the intact precursor mass. `_component_neutral_masses` is now shared with the channel-validity routine rather than recomputed |
| §7.3 no `a_charge`, no `frag_q` analogue | `a_charge` added to all 177 MASSter configs, and `masster_candidate_charge_feature` (default True) sums each candidate's Gasteiger charges — heavy-atom charges only, matching FraGNNet's `frag_q`. `atom_gasteiger_charges` is now attached to every molecular graph (0.03 ms/molecule; 1.1 s for all of NIST20), so the candidate feature does not depend on the atom feature list |
| §7.4 degenerate boundary embedding | `masster_boundary_bond_encoder` (default True) reuses FraGNNet's `CutBondEncoder` on the oriented (retained, lost) endpoints. `boundary_bond_endpoint_orientation` recovers the orientation the candidate builders do not emit, by looking both endpoints up in the sparse membership relation; it raises if a boundary bond does not have exactly one endpoint inside its component |
| §7.5 dead token columns | The intensity head drops the constant per-mask component count and the zeroed charge log-probability when its tokens are unique fragments. The legacy pathway-token mode keeps both, where they are live |
| §7.7 unread required field | `atom_formal_charges` removed from the `forward` field check. It is still attached to the graph and still unread |
| §7.8 shared-list mutation | `AtomOneHot.allowable_set` is a copy, so featurizing no longer appends `None` to `formula_utils.CANONICAL_ELEMENT_ORDER` |

Measured effect on the feature matrix, same 5,000-molecule NIST20 sample: constant atom columns
fall from **34/70 (49%) to 30/71 (42%)**. The corrected degree encoding leaves 8 constant columns
instead of 12, and those 8 are genuinely empty buckets — the sampled heavy-atom degrees are only
{1, 2, 3, 4}.

**Deliberately not changed:** §7.6, the fully constant `a_radical` (6 columns) and `a_chiral`
(5 columns). They are empty because NIST20 carries no radicals and its preprocessing strips
stereochemistry, not because the features are broken. Dropping them would make the feature set
dataset-specific and would silently discard signal on a corpus that does carry either. Removing
them is a config-only change if that trade is wanted.

Validation: full suite **3,890 passed, 1 skipped, 2 xfailed**; 20 new tests in
`tests/test_masster_candidate_features.py` and 9 in `tests/test_feat_utils.py` cover the
degeneracy, the orientation, the token width arithmetic and the new scalars. The lead merged
config resolves to a 71-dim atom input and a 1,285-dim intensity token and runs a finite
forward pass. Ruff is clean on the changed files apart from one pre-existing B007 in
`batch_mols_frags`. No training run, no evaluation, no cluster job.

Molecular preprocessing caches written before 2026-09-20 lack `atom_gasteiger_charges` and must
be regenerated.

## 12. GLACIER graph-feature parity (2026-09-22)

Three features were added to the shared registry — `a_fused_ring`, `b_ring_size`, `b_fused_ring` —
closing every graph-feature gap to GLACIER that carries signal on our data. The measured liveness
of each, the arms that enable them (MASSter `c4a_ring_topology`, FraGNNet
`graph_feature_parity`), and why `b_stereo` cannot be closed without a preprocessing change are
recorded once, in
[GLACIER feature parity](../notes/glacier_object_detection.md#closing-the-graph-feature-gap-2026-09-22).
The §7.6 finding above extends to bonds: NIST20, NIST23 and MassSpecGym all carry zero bond stereo
and zero chiral tags. *Corrected 2026-09-27:* this section earlier said nothing had been
trained, but both arms ran on 2026-09-23.

- MASSter `c4a_ring_topology` (`3gqgkl4x`) is null at −0.0027 against the 0.7270 control.
- FraGNNet `graph_feature_parity` (`m8qgi0ft`) crashed at epoch 77 and has not been read.

## 13. Complement composition in the candidate token (2026-09-22)

`masster_candidate_composition_features` (default **False**) adds the elemental composition
of each candidate's complementary neutral loss, as `log1p` counts over `CANONICAL_ELEMENT_ORDER`
plus one bucket for unlisted elements (18 columns). It is GLACIER's `psi(r_form - f_k)`; the
rationale, the measurement it rests on and the matched arm are recorded in
[the GLACIER note](../notes/glacier_object_detection.md#closing-the-head-input-gap-the-neutral-loss-complement-2026-09-22).

This is a genuine addition to §7.2 rather than a duplicate of it. §7.2 gave the candidate its own
mass; this gives the *complement* its composition. The precursor and component counts come from
one reduction (`_element_counts_by_group`), so the difference is exact by construction, and the
counting matches the channel-validity routine's treatment of hydrogen — explicit H atoms and the
implicit H count of every heavy atom both count.
