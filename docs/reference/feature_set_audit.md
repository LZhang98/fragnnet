# FraGNNet Feature Set Audit

> **Status:** REFERENCE &nbsp;·&nbsp; **Updated:** 2026-09-22
>
> Which atom/bond/fragment/conditioning features are wired. The 2026-06-09 body below is
> **stale on the "open gaps" line**: `a_ring_size` and `b_ring` are implemented and enabled in
> 44 FraGNNet configs, and three further features were added on 2026-09-22 — see
> [GLACIER feature parity](../notes/glacier_object_detection.md#closing-the-graph-feature-gap-2026-09-22)
> for the current gap list and the measured liveness of each feature.

Audited 2026-06-09 against branch `feiw_v2`. Covers atom/bond features, fragment node/edge features,
and spectrum-level conditioning inputs.

---

## Currently Used Features

### Atom node features (mol GNN) — all 8 enabled by default

| Name | RDKit call | Encoding | Notes |
|------|-----------|----------|-------|
| `a_onehot` | `GetSymbol()` | one-hot, 16 heavy + unknown | element type |
| `a_degree` | `GetDegree()`, `GetTotalDegree()` | continuous [deg, total_deg] + one-hot | total_degree − degree = num_Hs implicitly |
| `a_hybrid` | `GetHybridization()` | one-hot SP/SP2/SP3/SP3D/SP3D2 | |
| `a_formal` | `GetFormalCharge()` | one-hot [−2..2] | |
| `a_radical` | `GetNumRadicalElectrons()` | one-hot 0–4 | |
| `a_ring` | `IsInRing()`, `GetIsAromatic()` | 2 binary bits | **only is_in_ring + is_aromatic; no ring size. When used with `a_ring_size`, the is_in_ring bits are redundant — consider splitting into `a_aromatic` + dropping `a_ring`** |
| `a_mass` | `GetMass()` | continuous × 0.01 | |
| `a_chiral` | `GetChiralTag()` | one-hot CW/CCW/OTHER/UNSPECIFIED | **constant**: preprocessing strips stereochemistry, so every atom lands in the UNSPECIFIED bucket |

Off by default, available in the feature registry: `a_ring_size`, `a_num_hs`, `a_charge`
(Gasteiger), and `a_fused_ring` (atom in >=2 SSSR rings; added 2026-09-22, 6.5% of NIST20 atoms,
40.3% of molecules).

Optional positional encoding: `pyg_pe_embed_k` for random-walk PE (disabled by default).

### Bond edge features

| Name | Enabled | Notes |
|------|---------|-------|
| `b_degree` | always | bond type: SINGLE/DOUBLE/TRIPLE/AROMATIC |
| `b_stereo` | no | implemented, never enabled, and **measurably dead**: 0 of 127,249 sampled NIST20 bonds carry stereo because preprocessing strips it |
| `b_ring` | 44 configs | implemented (`bond.IsInRing()`); enabled in the new-mol-feats family and in every MASSter config |
| `b_conj` | 44 configs | same family as `b_ring` |
| `b_ring_size` | parity arm | added 2026-09-22; ring size membership `[in3, in4, in5, in6, inGe7]`, 50.3% of NIST20 bonds |
| `b_fused_ring` | parity arm | added 2026-09-22; bond in >=2 SSSR rings, 3.6% of NIST20 bonds |

### Fragment node features (default template)

| Name | Notes |
|------|-------|
| `cc` | mol_embedder atom embeddings pooled over fragment atoms |
| `base_formula` | fragment elemental formula (one-hot) |
| `depth` | depth in the DAG |
| `h_formulae_idx` | index into ±H-shifted formula list |
| `cmf_h_formulae_idx` | charge-migration H-formula index (requires `include_cmf: True`) |
| `boundary_pair` | Linear(cat(inside_atom_embed, outside_atom_embed)) over cut atoms — used in spectraverse configs, not in nist23 CV baseline |

Commented-out (implemented, rarely used): `h_counts`, `nb_iso_idx`, `prepass_rule_id`, `boundary_cc`.

#### `frag_q` — Gasteiger partial charge sum

`frag_q` is the sum of Gasteiger partial charges over all atoms in a fragment (shape `(num_nodes, 1)`).

**Storage:** unlike all other fragment node features, `frag_q` is **not packed into `x`** and is not registered in `node_feat_idxs`. It lives as a separate `dag_pyg.frag_q` attribute on the PyG Data object. This is because `x` is built around integer-encoded categorical features; inserting a continuous float column would require dtype changes throughout `get_node_feats`.

**Model use:** accessed directly as `frag_pyg.frag_q` in `FraGNNetModel.forward`, projected through `nn.Linear(1, embed_dim)` (`frag_q_proj`), and concatenated with other fragment node embeddings.

**When to refactor:** if a second continuous fragment node feature is added, introduce a `frag_x_cont` float tensor to unify them rather than adding another one-off attribute.

### Fragment edge features (default template)

| Name | Notes |
|------|-------|
| `cc` | atom embeddings at cut bonds |
| `base_formula` | formula constraint on edge |
| `h_range` | H-shift validity range; only used when `skip_edge_loss: False` |

### Spectrum-level conditioning

| Feature | Field | Newer configs | Notes |
|---------|-------|--------------|-------|
| NCE | `spec_nce` | yes | primary for FT/Orbitrap |
| ACE | `spec_ace` | spectraverse only | QTOF absolute eV; disabled in nist23 CV baseline |
| Precursor type | `spec_prec_type` | yes | adduct embedding |
| Instrument type | `spec_inst_type` | yes | FT/QTOF/IT/Other embedding |
| Frag mode | `spec_frag_mode` | spectraverse only | HCD/CID token; disabled in nist23 CV baseline |
| CE FiLM scaler | `ce_scale` | spectraverse only | affine output scaling conditioned on NCE/ACE stats |
| Frag-mode scaler | `frag_mode_scale` | spectraverse only | per-mode affine output scaling |

`spec_prec_mz` is loaded and passed through the batch but **not used as a model input** in FraGNNet.
It is only used post-hoc to convert formula probabilities back to m/z values.

---

## Genuine Gaps

### 1. Ring size — highest priority missing atom feature

`a_ring` has only 2 bits (is_in_ring, is_aromatic). There is no per-atom ring size signal.

Ring size directly controls fragmentation mechanism and energy threshold:
- 3/4-membered rings: ring strain → fragment at low CE
- 5/6-membered rings: stable; retro-Diels-Alder requires 6-membered rings specifically
- β-lactam (4-ring) is a signature fragment for antibiotics and many NPS
- 7+ membered rings: macrocyclic fragmentation patterns

Proposed `a_ring_size` feature: `[IsInRingOfSize(3), IsInRingOfSize(4), IsInRingOfSize(5), IsInRingOfSize(6), is_in_ring_ge7]` — 5 additional bits, zero inference overhead.

### 2. Bond ring membership (`b_ring`) — easy add

Already implemented in `feat_utils.py`, never enabled. Ring bonds vs. chain bonds have
fundamentally different fragmentation propensity. Adding it is a one-line config change plus
re-checking model input dimension.

### 3. Precursor m/z as conditioning input

`spec_prec_mz` never enters the model. The absolute precursor mass tells the model the scale of
the molecule: a 100 Da fragment from a 200 Da precursor is qualitatively different from the same
fragment from a 1000 Da precursor. Could be embedded as a normalized float alongside CE, using
the same `ce_insert_location` infrastructure.

### 4. Gasteiger partial charges

Not computed anywhere. Would give the model direct signal about charge-directed fragmentation
(protonation sites, charge-remote vs. charge-directed mechanisms). Requires a preprocessing pass
(`AllChem.ComputeGasteigerCharges(mol)`) and a new `a_charge` atom feature. High potential value
for charge-remote fragmentation modeling; notable engineering cost due to preprocessing changes.

---

## Things That Look Missing But Aren't

- **H count per atom**: `a_degree` stores both `GetDegree()` and `GetTotalDegree()` as continuous
  values; their difference equals `GetTotalNumHs()`. Not explicit, but learnable.
- **ACE for QTOF**: implemented and live in spectraverse configs; disabled in nist23 CV baseline.
- **Frag mode (HCD/CID)**: same — live in spectraverse, not in nist23 CV baseline.
- **boundary_pair**: implemented and live in spectraverse configs.

---

## Priority Order for Exploration

1. `a_ring_size` — zero preprocessing cost, directly relevant to fragmentation energetics,
   easy new `@register_atom_feat` class in `feat_utils.py`.
2. `b_ring` — already implemented, one-line config change.
3. Precursor m/z as model input — small embedding path in `base_model.py`, mirrors CE path.
4. Gasteiger charges (`a_charge`) — high value, but requires preprocessing pipeline changes.
