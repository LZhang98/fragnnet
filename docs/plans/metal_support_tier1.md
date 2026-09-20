# Tier 1 Metal Atom Support Plan

> **Status:** LANDED &nbsp;·&nbsp; **Updated:** 2026-04-26
>
> Tier-1 covalent organometallic support; shipped in `utils/frag_utils.py` (metal table) and `utils/data_utils.py` (MetalDisconnector skipped).

Support simple organometallics with covalent C–metal bonds (methylmercury, organotin, organolead)
through three targeted edits. Coordination compounds with dative bonds and ferrocene-type hapticity
complexes are explicitly out of scope.

## Scope

**In scope:** Molecules where the metal participates in standard covalent single bonds only
(C–Hg, C–Sn, C–Pb), overall neutral charge, no radical electrons.

**Out of scope:** Coordination complexes with dative bonds (Fe-porphyrins, cisplatin-type),
ionic metal salts, multi-fragment metal species, η-bonded sandwich complexes (ferrocene).

---

## Change 1 — Add metals to element whitelist

**File:** `src/fragnnet/utils/frag_utils.py` line 51

Add Hg, Sn, Pb to `ELEMENT_TO_VE`:

```python
ELEMENT_TO_VE = {
    "C": 4,
    "O": 2,
    "N": 3,
    "P": 3,
    "S": 2,
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "Se": 2,
    "Si": 4,
    "As": 3,
    "B": 3,
    # Tier 1 metals — covalent organometallics only (C–metal single bonds)
    "Hg": 2,   # methylmercury, ethylmercury
    "Sn": 4,   # organotin (tributyltin, trimethyltin)
    "Pb": 4,   # organolead (tetraethyllead)
}
```

**Auto-updates for free (no additional edits needed):**
- `HEAVY_ELEMENTS`, `NUM_HEAVY_ELEMENTS`, `ELEMENTS`, `NUM_ELEMENTS` — all derived from the dict
- `CANONICAL_ELEMENT_ORDER`, `ELEMENT_TO_IDX`, `IDX_TO_ELEMENT` — all derived
- Formula arrays (per-element count vectors) — sized from `NUM_HEAVY_ELEMENTS`
- Mol encoder atom one-hot (`AtomOneHot.allowable_set`) — reads `CANONICAL_ELEMENT_ORDER`
- All preprocessing `num_out_set_elements == 0` filters — reference `frag_utils.ELEMENTS`

---

## Change 2 — Fix `Cleanup()` metal disconnection

**File:** `src/fragnnet/utils/data_utils.py` line 153

`rdMolStandardize.Cleanup()` internally runs `MetalDisconnector`, which strips covalent
metal bonds. Replace it with the two chemically meaningful sub-steps, omitting the disconnector:

```python
# Before:
mol = rdMolStandardize.Cleanup(mol)

# After:
mol = rdMolStandardize.Normalize(mol)    # normalize functional groups
mol = rdMolStandardize.Reionize(mol)     # canonical ionization state
```

**Verify with a quick sanity check after this change:**
```python
from fragnnet.utils.data_utils import mol_from_smiles
mol = mol_from_smiles("[CH3][Hg]Cl")   # methylmercury chloride
assert mol is not None
assert mol.GetNumAtoms() == 3          # C, Hg, Cl — not disconnected
```

---

## Change 3 — Update `num_elements` in configs

**File:** `config/models_template` line 156

`num_elements` is the count of heavy elements used in formula embeddings. It is hardcoded
in YAML and must match `NUM_HEAVY_ELEMENTS` in `frag_utils.py` at runtime.

The baseline whitelist had 13 heavy elements (not 12 — the config was already stale by one).
Adding 3 metals brings the total to **16**.

```bash
# Verify current count
python -c "from fragnnet.utils.frag_utils import NUM_HEAVY_ELEMENTS; print(NUM_HEAVY_ELEMENTS)"
# Should print 16
```

Update `config/models_template`:
```yaml
num_elements: 16  # Number of heavy elements (13 base + 3 Tier-1 metals: Hg, Sn, Pb)
```

**This is a checkpoint-breaking change.** The formula embedder dimension changes, so
existing saved checkpoints will not load. Options:
- Retrain from scratch (cleanest)
- Zero-pad the formula embedding weight matrix for new metal dimensions and fine-tune
  (saves GNN weights, but requires manual checkpoint surgery)

---

## Validation

### Unit tests

See `tests/test_metal_support.py`. Covers:
- End-to-end featurization for methylmercury, triethyltin, tetraethyllead
- Metal elements present in element counts
- C–Hg bond not disconnected by standardization
- Whitelist contains all three metals
- `NUM_HEAVY_ELEMENTS` matches `len(ELEMENT_TO_VE)`

### Manual check

Confirm a known organomercury SMILES runs through the full DAG preparation pipeline
(`preproc_scripts/03_prepare_dag_feats.py`) without crashing, and that fragment
formulas include the metal element correctly.

---

## Assumptions and guards

- **Charge assert** (`frag_utils.py:237`): neutral organometallics have zero molecule-level
  charge; this assert will not fire. Ionic species are out of scope for Tier 1.
- **Radicals assert** (`frag_utils.py:292`): closed-shell organometallics have no unpaired
  electrons; this assert will not fire.
- **Bond types** (`BOND_TYPE_TO_IDX`): covalent C–metal bonds are SINGLE; no new bond type
  handling needed.
- **H-transfer model**: the ±4H BFS transfer is meaningless on metal atoms but harmless —
  it generates physically unrealistic fragments that won't match real peaks and will simply
  contribute zero loss.

---

## Effort estimate

| Step | File | Time |
|------|------|------|
| Add metals to `ELEMENT_TO_VE` | `frag_utils.py:51` | 15 min |
| Replace `Cleanup()` | `data_utils.py:153` | 30 min |
| Update `num_elements` in all configs | `config/models_template` | 30 min |
| Write and run tests | `tests/test_metal_support.py` | 30 min |
| **Total** | | **~1.5 hours** |
