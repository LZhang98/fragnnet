# DAG Deduplication Strategies

Post-processing step that removes redundant fragment nodes from DAG tar files before
training. Redundancy arises when the same heavy-atom formula appears at multiple depths
in the fragmentation DAG; keeping duplicates wastes memory and can bias the model toward
over-represented formulas.

## Strategies

### leaf_only
Remove max-depth (leaf) nodes whose **base formula** already appears at a strictly
shallower depth. Conservative: only touches true leaves, never mid-DAG nodes.

### all_depths
Remove any node (at any depth) whose **base formula** already appears at a strictly
shallower depth. More aggressive than `leaf_only`.

### hybrid *(recommended)*
Remove a deeper node only when **all its H-transfer formula indices** are already covered
by shallower nodes of the same base formula. Guarantees 0% H-formula recall loss while
still aggressively deduplicating. Preserves minimum-depth structure.

### h_coverage
Remove any node (same or deeper depth) whose H-index set is a subset of already-seen
nodes of the same base formula. Most aggressive; deduplicates same-depth duplicates too.
Near-zero H-recall loss in practice.

## H-formula Recall

Fraction of spectrum-matched H-transfer formula indices that survive deduplication,
broken down by maximum allowed |H-delta|.

### D3 (nist20_ma_mi_d3_h4_isoFalse_smpFalse, full dataset)

| h_max     | baseline | leaf_only | all_depths | hybrid   | h_coverage |
|-----------|----------|-----------|------------|----------|------------|
| h_max=1   | 100.00%  | 97.14%    | 96.65%     | 100.00%  | 99.99%     |
| h_max=2   | 100.00%  | 96.31%    | 95.73%     | 100.00%  | 99.99%     |
| h_max=3   | 100.00%  | 95.22%    | 94.53%     | 100.00%  | 99.99%     |
| h_max=4   | 100.00%  | 93.24%    | 92.48%     | 99.9992% | 99.98%     |
| all       | 100.00%  | 93.24%    | 92.48%     | 99.9992% | 99.98%     |

### D4 (nist20_ma_mi_d4_h4_isoFalse_smpFalse, full dataset)

| h_max     | baseline | leaf_only | all_depths | hybrid   | h_coverage |
|-----------|----------|-----------|------------|----------|------------|
| h_max=1   | 100.00%  | 96.98%    | 93.27%     | 99.9983% | 99.98%     |
| h_max=2   | 100.00%  | 95.89%    | 91.67%     | 99.9985% | 99.98%     |
| h_max=3   | 100.00%  | 94.11%    | 89.22%     | 99.9982% | 99.97%     |
| h_max=4   | 100.00%  | 90.94%    | 85.79%     | 99.9963% | 99.94%     |
| all       | 100.00%  | 90.94%    | 85.79%     | 99.9963% | 99.94%     |

**Takeaway:** `hybrid` is the safe default — zero recall loss on D3, negligible (0.004%)
on D4. `h_coverage` trades a tiny recall loss for slightly more aggressive dedup.
`leaf_only` and `all_depths` both drop meaningful recall at h_max=4 (6–14% on D4).

## Output Files

Tars are written to `data/frag/dedup/` with the suffix appended to the base name:

```
data/frag/dedup/
  nist20_ma_mi_d3_h4_isoFalse_smpFalse_{strategy}.tar   # D3 variants
  nist20_ma_mi_d4_h4_isoFalse_smpFalse_{strategy}.tar   # D4 variants
  nist20_ma_mi_d3_h4_isoFalse_smpFalse_dedup_recall.csv
  nist20_ma_mi_d4_h4_isoFalse_smpFalse_dedup_recall.csv
```

The internal tar directory name matches the tar filename (e.g.
`nist20_ma_mi_d3_h4_isoFalse_smpFalse_hybrid/dags/`).

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_dag_dedup_compare.py` | Create all 4 strategy tars in parallel + compute recall table |
| `scripts/run_dag_postproc_dedup.py` | Create a single dedup tar (all_depths strategy) |
| `scripts/run_dag_dedup_dir.py` | Apply dedup to a directory-format dataset (no tar) |
| `scripts/debug/debug_dag_postproc_dedup.py` | Sample-based analysis of all strategies (read-only) |

### Reproducing the dedup tars

```bash
# D3
python scripts/run_dag_dedup_compare.py \
    --input_tar data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse.tar \
    --out_dir   data/frag/dedup \
    --strategies leaf_only all_depths hybrid h_coverage \
    --recall_sample 0

# D4
python scripts/run_dag_dedup_compare.py \
    --input_tar data/frag/nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar \
    --out_dir   data/frag/dedup \
    --strategies leaf_only all_depths hybrid h_coverage \
    --recall_sample 0
```
