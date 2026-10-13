# Data Mapping Definitions

> **Status:** REFERENCE · **Updated:** 2026-10-06
>
> Weighted CE input contract updated; remaining historical schema should be verified against code.

## Input Batch Dictionary Reference

```python
batched_mol_frag = {
    "mol_pyg": mol_pyg, # PyG batch for molecules
    "frag_pyg": frag_pyg, # PyG batch for fragments
    "mol_num_nodes": mol_num_nodes, # Cumulative node count for molecules
    "frag_num_nodes": frag_num_nodes, # Cumulative node count for fragments
    "frag_formula_peak_idxs": frag_formula_peak_idxs,  # Flattened tensor mapping each peak to its local formula index (0-based per molecule)
    "frag_formula_peak_mzs": frag_formula_peak_mzs, # Flattened tensor of m/z values for each peak
    "frag_formula_peak_probs": frag_formula_peak_probs, # Flattened tensor of theoretical isotopic probabilities for each peak
    "frag_formula_sizes": frag_formula_sizes, # Tensor containing the number of unique formulas (fragments) with peaks for each molecule
    "frag_formula_cumsizes": frag_formula_cumsizes, # Cumulative sum of frag_formula_sizes (offsets for formula indices in batch)
    "frag_formula_peak_sizes": frag_formula_peak_sizes, # Tensor containing the total number of peaks (isotopes) for each molecule
}
```

## Optional weighted CE inputs

With `spec_params.merge_ce_aggregation: weighted`, merged batches include one-dimensional
`spec_nce_weights` and `spec_ace_weights`, aligned with the flattened `spec_nce` and `spec_ace`
readings and their respective batch indices. `spec_ce_weights` aliases the configured primary
CE signal's weights. Each spectrum has unit total weight separately for NCE and ACE.

The merger weights primary-CE schedules equally, then their replicates equally, then readings
within each row equally. Both models use these weights when pooling CE embeddings. Collated
CE statistics use weighted mean and standard deviation over valid readings; missing values keep
the existing missing-token semantics. Legacy batches omit weight fields and retain mean pooling.
See the [CE correction record](../findings/masster/masster_context_design_audit.md) for constraints
and compatibility behavior.

## Output Dictionary Reference

Definitions:
*   $N$: Total number of nodes in batch
*   $F$: Total number of formulae in batch
*   $B$: Total number of spectra in batch
*   $P$: Total number of peaks in batch
*   $J$: Total number of valid (non-null) joint (node, formula) pairs
*   $E$: Total number of edges in fragment graphs
*   $H$: Number of hydrogen counts considered ($2 \times \text{num\_hs} + 1$)
*   $N'$: Total number of unique neighborhoods (if `nb_iso` is True)
*   $J'$: Total number of valid joint (neighborhood, formula) pairs (if `nb_iso` is True)

```python
out_d = {
    "pred_mzs": spec_mzs, # Predicted m/z values for the spectrum peaks [P]
    "pred_logprobs": spec_logprobs, # Predicted log probabilities (intensities) for the spectrum peaks [P]
    "pred_batch_idxs": spec_batch_idxs, # Batch indices mapping each peak to a spectrum in the batch [P]
    "pred_formula_logprobs": frag_real_formula_logprobs, # Log probabilities of formulas P(f) [F]
    "pred_formula_formula_idxs": frag_real_formula_formula_idxs, # Indices of the formulas corresponding to pred_formula_logprobs [F]
    "pred_formula_batch_idxs": frag_formula_batch_idxs, # Batch indices for the formulas [F]
    "pred_node_logprobs": frag_real_node_logprobs, # Log probabilities of nodes (fragments) P(n) [N]
    "pred_node_node_idxs": frag_real_node_node_idxs, # Indices of the nodes corresponding to pred_node_logprobs [N]
    "pred_node_batch_idxs": frag_node_batch_idxs, # Batch indices for the nodes [N]
    "pred_node_formula_logprobs": frag_real_node_formula_logprobs, # Conditional log probabilities P(f|n) [J]
    "pred_formula_node_logprobs": frag_real_formula_node_logprobs, # Conditional log probabilities P(n|f) [J]
    "pred_joint_logprobs": frag_real_joint_logprobs, # Joint log probabilities P(f,n) [J]
    "pred_joint_node_idxs": frag_real_joint_node_idxs, # Node indices for the joint distribution entries [J]
    "pred_joint_formula_idxs": frag_real_joint_formula_idxs, # Formula indices for the joint distribution entries [J]
    "pred_joint_batch_idxs": frag_real_joint_batch_idxs, # Batch indices for the joint distribution entries [J]
    "pred_joint_h_counts": frag_real_joint_h_counts, # Hydrogen counts for the joint distribution entries [J]
    "pred_joint_h_idxs": frag_real_joint_h_idxs, # Indices for hydrogen counts in joint distribution [J]
    "pred_null_formula_logprob": frag_null_formula_logprob, # Log probability of the NULL formula (unexplained intensity) [B]
    "pred_edge_logprobs": frag_edge_logprobs, # Log probabilities for edges (if edge loss is used) [E]
    "pred_edge_h_diffs": frag_edge_h_diffs, # Hydrogen count differences across edges [E, H*H]
    "pred_edge_h_range_masks": frag_edge_h_range_masks, # Masks for valid hydrogen ranges on edges [E, H*H]
    "pred_edge_h_logprobs": frag_edge_h_logprobs, # Log probabilities for hydrogen transfers on edges [E, H*H]
    "pred_edge_batch_idxs": frag_edge_batch_idxs, # Batch indices for edges [E]
    "pred_oos_logprobs": oos_logprobs, # Log probability of Out-Of-Scope (OOS) intensity [B]
    "pred_h_counts": frag_real_h_counts, # Hydrogen counts for the hydrogen distribution [B*H]
    "pred_h_batch_idxs": frag_real_h_batch_idxs, # Batch indices for the hydrogen distribution [B*H]
    "pred_h_logprobs": frag_real_h_logprobs, # Log probabilities of hydrogen counts [B*H]
    "pred_nb_node_logprobs": frag_nb_node_logprobs, # (Isomorphism) Log probabilities of neighborhood nodes P(n') [N']
    "pred_nb_node_formula_logprobs": frag_nb_node_formula_logprobs, # (Isomorphism) Conditional log probabilities P(f|n') [J']
    "pred_nb_formula_node_logprobs": frag_nb_formula_node_logprobs, # (Isomorphism) Conditional log probabilities P(n'|f) [J']
    "pred_nb_node_node_logprobs": frag_nb_node_node_logprobs, # (Isomorphism) Conditional log probabilities P(n|n') [N]
    "pred_nb_node_node_idxs": frag_nb_node_node_idxs, # (Isomorphism) Indices for node-node mapping [N]
    "pred_nb_node_batch_idxs": frag_nb_node_batch_idxs, # (Isomorphism) Batch indices for neighborhood nodes [N']
    "pred_nb_joint_logprobs": frag_nb_joint_logprobs, # (Isomorphism) Joint log probabilities P(f,n') [J']
    "pred_nb_joint_node_idxs": frag_nb_joint_node_idxs, # (Isomorphism) Node indices for joint distribution [J']
    "pred_nb_joint_formula_idxs": frag_nb_joint_formula_idxs, # (Isomorphism) Formula indices for joint distribution [J']
    "pred_nb_joint_batch_idxs": frag_nb_joint_batch_idxs, # (Isomorphism) Batch indices for joint distribution [J']
    "pred_nb_node_node_node_idxs": frag_nb_node_node_node_idxs, # (Isomorphism) Node indices for node-node mapping [N]
    "pred_nb_node_node_batch_idxs": frag_nb_node_node_batch_idxs, # (Isomorphism) Batch indices for node-node mapping [N]
}
```

## 1. Conditional Probability $P(n|f)$

$$ \log P(n|f) = \log P(f,n) - \log P(f) $$

Aligned arrays of shape `[N*F]`:
*   `pred_formula_node_logprobs`: $\log P(n|f)$
*   `pred_joint_formula_idxs`: Formula indices corresponding to $f$
*   `pred_joint_batch_idxs`: Batch indices corresponding to the spectrum

## 2. Marginal Formula Probability $P(f)$

$$ P(f) = \sum_{n} P(f,n) $$

Aligned arrays of shape `[F]`:
*   `pred_formula_logprobs`: $\log P(f)$
*   `pred_formula_batch_idxs`: Batch indices corresponding to the spectrum
*   `pred_formula_formula_idxs`: Formula indices corresponding to $f$

## 3. Marginal Node Probability $P(n)$

$$ P(n) = \sum_{f} P(f,n) $$

Aligned arrays of shape `[N]`:
*   `pred_node_logprobs`: $\log P(n)$
*   `pred_node_batch_idxs`: Batch indices corresponding to the spectrum
*   `pred_node_node_idxs`: Node indices corresponding to $n$

## 4. Peak Probability $P(p)$

$$ \log P(p) = \log P(f)_{\text{adjusted}} + \log P(p|f) $$

where $P(f)_{\text{adjusted}}$ accounts for Out-Of-Scope probability.

Aligned arrays of shape `[P]`:
*   `pred_logprobs`: $\log P(p)$ (Final peak intensity)
*   `pred_mzs`: m/z value of the peak
*   `pred_batch_idxs`: Batch indices corresponding to the spectrum

**Note:** Shape `[P]` is generally larger than `[F]` because one formula $f$ generates multiple isotopic peaks $p$.

---

# Access Steps

## Step 1: Select Spectrum
Select which spectrum you want (let's say `batch_idx` of $k$).

```python
b_pred_formula_node_mask = (b_joint_batch_idxs == k)
b_pred_formula_node_logprobs = pred_formula_node_logprobs[b_pred_formula_node_mask]
b_pred_joint_formula_idxs = pred_joint_formula_idxs[b_pred_formula_node_mask]

b_pred_formula_mask = (pred_formula_batch_idxs == k)
b_pred_formula_logprobs = pred_formula_logprobs[b_pred_formula_mask]
b_pred_formula_formula_idxs = pred_formula_formula_idxs[b_pred_formula_mask]

b_pred_node_mask = (pred_node_batch_idxs == k)
b_pred_node_logprobs = pred_node_logprobs[b_pred_node_mask]
b_pred_node_node_idxs = pred_node_node_idxs[b_pred_node_mask]
```

## Step 2: Select Formula
Figure out which formula you want (let's say `formula_idx` = $j$).
*Note: this assumes that `formula_idx` $j$ is actually in the spec at `batch_idx` $k$.*

```python
bf_pred_formula_node_mask = (b_pred_joint_formula_idxs == j)
# ...

bf_pred_formula_mask = (b_pred_formula_formula_idxs == j)
# ...
```

## Step 3: Select Node
Figure out which node you want (let's say $i$).
*Note: this assumes that `node_idx` $i$ is in `batch_idx` $k$ and has nonzero probability for `formula_idx` $j$.*

```python
# To get P(n|f) for specific node i and formula j (from step 2 context)
bf_pred_formula_node_mask = (b_pred_joint_formula_idxs == j)
bf_pred_joint_node_idxs = b_pred_joint_node_idxs[bf_pred_formula_node_mask]
bf_pred_formula_node_logprobs = b_pred_formula_node_logprobs[bf_pred_formula_node_mask]

node_mask = (bf_pred_joint_node_idxs == i)
p_n_given_f = bf_pred_formula_node_logprobs[node_mask]

# To get P(n) (marginal)
node_mask_marginal = (b_pred_node_node_idxs == i)
p_n = b_pred_node_logprobs[node_mask_marginal]

## Step 4: Map Formulas to Peaks
To see which peaks correspond to a formula $f$ (index `f_idx`):

```python
# Note: frag_formula_peak_idxs contains LOCAL formula indices (per molecule)
# You need the input tensor 'frag_formula_peak_idxs' from the batch data

# Get all peaks generated by formula f_idx
peak_indices = th.nonzero(frag_formula_peak_idxs == f_idx).flatten()

for peak_idx in peak_indices:
    peak_mz = pred_mzs[peak_idx]
    peak_intensity = pred_logprobs[peak_idx].exp()
    print(f"Formula {f_idx} -> Peak at {peak_mz:.4f} m/z with intensity {peak_intensity:.4f}")
```
```

## Split ratios (from local memory, re-measured 2026-10-06)

- `preproc_scripts/04_prepare_split.py` defaults: `--test_frac 0.2 --val_frac 0.2 --split_key
  inchikey_s --split_type random --meta_rseed 42`. Both fractions come off the total key count
  (test first, then val), so train is the 0.6 residual.
- Spectra (train / val / test): `nist20_ma_mi_inchikey` 287,721 / 96,153 / 96,128;
  `nist20_inchikey` 157,678 / 52,020 / 52,621.
- Scaffold splits do not hold the ratio: `nist20_ma_mi_scaffold` is 0.547 / 0.302 / 0.151
  (262,693 / 144,775 / 72,534). Do not quote 60/20/20 for a scaffold run. No MASSter config uses a
  scaffold split.
