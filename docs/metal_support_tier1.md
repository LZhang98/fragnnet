# Tier 1 Metal Atom Support Plan

Support simple organometallics with covalent C–metal bonds (e.g., methylmercury, organotin,
organoplatinum) through three targeted edits. Coordination compounds with dative bonds are
explicitly out of scope.

## Scope

**In scope:** Molecules where the metal participates in standard covalent single bonds only
(C–Hg, C–Sn, C–Pb, C–Pt, etc.), overall neutral charge, no radical electrons.

**Out of scope:** Coordination complexes with dative bonds (Fe-porphyrins, cisplatin-type),
ionic metal salts, multi-fragment metal species.

---

## Change 1 — Add metals to element whitelist

**File:** `src/fragnnet/utils/frag_utils.py` line 51

Add target metals to `ELEMENT_TO_VE`. Start with the most common in MS/MS databases:

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
    # Tier 1 metals — covalent organometallics only
    "Hg": 2,   # methylmercury, ethylmercury
    "Sn": 4,   # organotin (tributyltin, trimethyltin)
    "Pb": 4,   # organolead (tetraethyllead)
    "Pt": 4,   # organoplatinum drugs (oxaliplatin, carboplatin)
    "Fe": 2,   # ferrocenes (covalent cyclopentadienyl bonds)
}
```

Adjust the list to match which databases are being targeted.

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
metal bonds. Replace it with the individual sub-steps, omitting the disconnector:

```python
# Before:
mol = rdMolStandardize.Cleanup(mol)

# After:
mol = rdMolStandardize.Normalize(mol)    # normalize functional groups
mol = rdMolStandardize.Reionize(mol)     # canonical ionization state
```

`Normalize` and `Reionize` are the two chemically meaningful steps for ML standardization.
`MetalDisconnector` and `LargestFragmentChooser` are the steps we're dropping — the latter
was already commented out at line 159.

**Verify with a quick sanity check after this change:**
```python
from fragnnet.utils.data_utils import mol_from_smiles
mol = mol_from_smiles("[CH3][Hg]Cl")   # methylmercury chloride
assert mol is not None
assert mol.GetNumAtoms() == 3          # C, Hg, Cl — not disconnected
```

---

## Change 3 — Update `num_elements` in configs

**File:** `config/template.yml` line 156

`num_elements` is the count of heavy elements used in formula embeddings. It is hardcoded
in YAML and must match `NUM_HEAVY_ELEMENTS` in `frag_utils.py` at runtime.

```bash
# Find all configs that set this field
grep -rn "num_elements:" config/
```

Update every occurrence from `12` to the new count (e.g., `17` if 5 metals are added).

**This is a checkpoint-breaking change.** The formula embedder dimension changes, so
existing saved checkpoints will not load. Options:
- Retrain from scratch (cleanest)
- Zero-pad the formula embedding weight matrix for new metal dimensions and fine-tune
  (saves GNN weights, but requires manual checkpoint surgery)

---

## Validation

### Unit test — end-to-end featurization

Add to `tests/test_metal_support.py`:

```python
import pytest
from fragnnet.utils.frag_utils import mol_to_frag_data, ELEMENT_TO_VE

METAL_SMILES = [
    ("[CH3][Hg]Cl",    "methylmercury chloride"),
    ("CC[Sn](CC)(CC)Cl", "triethyltin chloride"),
]

@pytest.mark.parametrize("smiles,name", METAL_SMILES)
def test_metal_featurization(smiles, name):
    """Metal atoms should featurize without error and appear in element counts."""
    mol_d = mol_to_frag_data(smiles)
    assert mol_d is not None, f"featurization failed for {name}"

@pytest.mark.parametrize("smiles,name", METAL_SMILES)
def test_metal_in_element_counts(smiles, name):
    """Metal element should appear in element_counts with count > 0."""
    mol_d = mol_to_frag_data(smiles)
    metal_sym = [a for a in mol_d["elems"] if a not in {"C","H","O","N","P","S","F","Cl","Br","I","Se","Si","As","B"}]
    assert len(metal_sym) > 0, f"no metal atom found in element_counts for {name}"

def test_metal_not_disconnected():
    """Cleanup() fix must not disconnect C–Hg bond."""
    from fragnnet.utils.data_utils import mol_from_smiles
    mol = mol_from_smiles("[CH3][Hg]Cl")
    assert mol.GetNumAtoms() == 3   # C, Hg, Cl intact
    assert mol.GetNumBonds() == 2   # C-Hg and Hg-Cl
```

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
| Update `num_elements` in all configs | `config/**/*.yml` | 30 min |
| Write and run tests | `tests/test_metal_support.py` | 30 min |
| **Total** | | **~1.5 hours** |
