# MASShatter Feature Set Audit

> **Status:** REFERENCE &nbsp;·&nbsp; **Updated:** 2026-09-20
>
> Which atom/bond/candidate/conditioning features MASShatter actually consumes.
> Headline: **34 of 70 atom-feature columns are constant on NIST20**, 12 of them because
> `a_degree`'s one-hot is mis-encoded; the candidate intensity head is mass-blind.

Audited against `31c487db` on branch `research/masshatter-ion-h-selection`, over all 175
`model_type: mass_shatter` configs under `config/`. Column-constancy measured on a 5,000-molecule
sample (seed 0) of `data/proc/nist20/mol_df.pkl` (120,641 atoms, 254,498 bonds).
The companion FraGNNet document is [feature_set_audit.md](feature_set_audit.md); the metadata
conditioning design is audited separately in
[masshatter_context_design_audit.md](../findings/masshatter_context_design_audit.md).

---

## 1. Where features enter

MASShatter consumes features at five sites. Everything else is derived.

| Site | Code | Inputs |
|------|------|--------|
| Precursor GNN | `GNN` in [nn_blocks.py](../../src/fragnnet/model/nn_blocks.py) | `mol_pyg.x` (atom feats), `mol_pyg.edge_attr` (bond feats), optionally `context` |
| Bond break head | `BondFocusedBreakHead` / `EndpointBondBreakHead` | endpoint atom embeddings, `chemical_bond_attr`, `context` |
| Charge affinity head | `AtomChargeAffinityHead` | atom embeddings, `context` |
| Candidate intensity head | `CandidateIntensityTransformerHead` | 3 pooled embeddings + `context` + 3 scalars (+2 optional) |
| Context encoder | `MassShatterContextEncoder` | adduct, instrument, frag mode, NCE, ACE |

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
| `a_charge` | 1 | **no** | — | Gasteiger partial charge. Implemented, used by FraGNNet's `pcdl_nist23*`/`nps_pcdl_nist23*` configs, **never enabled in any MASShatter config** (see §7.3) |
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
(`mol_hidden_size: 384`, `mass_shatter_context_dim: 128`) the token input is **1,283 dims**:

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

`use_base_probability_feature = not mass_shatter_proposal_anchored`, and 135/175 configs set
`proposal_anchored: True`, so both base-probability inputs are off in the current arms.

What is **not** in the token: fragment mass, channel m/z, elemental composition, hydrogen count,
RDBE, precursor m/z or mass fraction, partial charge, and the cut bonds' own feature vectors.
Extent is represented only by `log1p(atom count)` and `cut_counts`; pooling is `mean`, so the
size signal in the embeddings is normalized away.

## 5. Spectrum-level conditioning

Encoded once per spectrum by `MassShatterContextEncoder`, then broadcast to bonds, atoms and
candidates. Unlike FraGNNet there is no separate insert-location machinery — one vector, all heads.

| Feature | Field | Configs enabling it | Notes |
|---------|-------|---------------------|-------|
| NCE | `spec_nce` | 175/175 | `ce_insert_type: embed` in 152; learned embedding, mean-pooled over merged CE readings |
| ACE | `spec_ace` | 24/175 | dual-CE arms only |
| Precursor adduct | `spec_prec_type` | 175/175 | required; `prec_type: True` is enforced at setup |
| Instrument | `spec_inst_type` | 79 explicit True, 93 inherit template True | empty vocabulary skips the embedding entirely |
| Frag mode | `spec_frag_mode` | 77/175 | template default False; one extra row for unknown |
| Precursor m/z | `spec_prec_mz` | **0** | loaded and batched, used only by metrics — never a model input (same gap as FraGNNet) |
| SMARTS rule id | `rule_indices` | 2/175 | `SmartsMechanismHead` rule embedding; `mass_shatter_smarts_templates` defaults False |

`context_in_gnn` (context appended to atom inputs) is on in 3/175 configs; it is rejected at
construction together with `share_proposal_across_ev`.

## 6. Non-learned chemistry fields

Attached by `MolGraph.get_pyg_graph` and required by `MassShatterModel.forward`. These are
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
them keeps `node_feature_dim = 70` and every checkpoint tensor shape. It still changes input
*semantics*, so an existing checkpoint loaded against a fixed featurizer would silently see
different inputs — per [engineering rules §6](../rules/engineering.md) this needs a new feature
name or an opt-in flag rather than an in-place fix.

### 7.2 The candidate intensity head is mass-blind

No m/z, fragment mass, or composition enters any learned parameter (§4). Masses exist in scope:
`ion_mzs` is computed at `mass_shatter_model.py:2264`, before the intensity head is called at
`:2402`. Feeding fragment mass, mass fraction of the precursor, or per-channel m/z into the token
is a local change to an already-computed tensor.

### 7.3 `a_charge` is never enabled for MASShatter, and MASShatter is the model that wants it

MASShatter has an explicit `AtomChargeAffinityHead` whose whole job is deciding which component
keeps the charge, plus a CRF/CMF ion-mode split. Its atom inputs contain no electrostatic prior.
Gasteiger charges are implemented, computed on the fly (no DAG regeneration), and already used by
FraGNNet's newest configs, which also carry `frag_q` (the per-fragment Gasteiger sum). MASShatter
has no `frag_q` analogue in its candidate token.

### 7.4 Cut-bond identity is discarded before intensity scoring

`boundary_embeddings` pool `(atom_emb[u] + atom_emb[v]) / 2`, so `b_degree`/`b_ring`/`b_conj` for
the *severed* bonds reach the intensity head only through whatever GINE mixed into the endpoint
atom states. Meanwhile `BondFocusedBreakHead` (3 layers in 156/175 configs) builds refined
per-bond states that do see `chemical_bond_attr` directly — and discards them after emitting
break logits. Pooling those existing bond states as the boundary source is shape-local.

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

`MassShatterModel.forward` fails a batch whose graphs lack `atom_formal_charges`, yet no model or
chemistry code reads it (repo-wide: written in `feat_utils.py:257`, listed in the required-field
check at `mass_shatter_model.py:1904`, asserted in `tests/test_spec_mol_dataset.py`). Formal charge
is also absent from the fragment-mass arithmetic, which uses atomic numbers plus hydrogen counts.

### 7.8 The featurizer mutates the shared canonical element order

`AtomOneHot.allowable_set` *is* `frag_utils.CANONICAL_ELEMENT_ORDER` (same list object), and
`one_hot_encoding(..., encode_unknown=True)` appends `None` to it on first use. After featurizing
any molecule the shared list is 18 long instead of 17. Harmless today — its two consumers do
`if elem in elem_d`, and `ELEMENT_TO_IDX`/`IDX_TO_ELEMENT` are built at import — but any
length-based or lazily-built consumer would break.

## 8. Config drift

| Group | Count | Feature shape |
|-------|-------|---------------|
| Experiment configs under `config/masshatter/**` and `config/sweeps_merged/masshatter/**` | 169 | 70 atom / 9 bond |
| `config/nist20_merged_inchikey/mass_shatter/s0.yml` and 5 `config/debug/debug_tiny_mass_shatter*.yml` | 6 | template 60 atom / 5 bond |

The canonical seeded config (`nist20_merged_inchikey/mass_shatter/s0.yml`) therefore does not
match the feature set every experiment arm actually trains with, and produces a different
`node_feature_dim`. `config/template.yml` keeps `a_ring_size`, `a_charge`, `b_ring` and `b_conj`
commented out.

Minor: `mass_shatter_pl.py:967-969` reads the three Graphormer structural keys with
`self.hparams.get(...)`, which [Python rules](../rules/python.md) forbid for hyperparameters;
all three are declared in `config/template.yml`, so the fallback only hides missing keys in
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
[regularization sweep plan](../plans/masshatter_regularization_sweep_plan.md).
