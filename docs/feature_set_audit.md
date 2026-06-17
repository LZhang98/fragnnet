# FraGNNet Feature Set Audit

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
| `a_chiral` | `GetChiralTag()` | one-hot CW/CCW/OTHER/UNSPECIFIED | |

Optional positional encoding: `pyg_pe_embed_k` for random-walk PE (disabled by default).

### Bond edge features

| Name | Enabled | Notes |
|------|---------|-------|
| `b_degree` | always | bond type: SINGLE/DOUBLE/TRIPLE/AROMATIC |
| `b_stereo` | no | implemented, never enabled |
| `b_ring` | no | implemented (`bond.IsInRing()`), never enabled |
| `b_conj` | no | implemented, never enabled |

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
