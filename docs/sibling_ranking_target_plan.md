# Sibling-ranking supervision on P(n)

**Status 2026-07-22: coverage gate PASSED, loss not yet implemented.**

Supervise the *within-sibling-set ordering* of the existing node marginal `P(n)` using per-spectrum
targets mined from observed peaks. No new parameters, no DAG change, no edge machinery.

## 1. Why this and not a competition head

The Stage 0 result (`project_stage0_branching_headroom_model`) — model dominant-sibling mismatch
~45–50% against a ~0.11 replicate floor — was read as evidence for **edge-level** competition, which
motivated FluxMPNN Stage 3 and the β-supervision line. That reading is wrong.

In `scripts/eval/stage0_branching_model_vs_oracle.py`, *both* sides of the comparison are node
marginals:

- `q_base` = `P(n)` restricted to a parent's children, renormalized
- `q_obs`  = observed peak intensity over the same children, renormalized

Neither is an edge quantity. The metric is valid and unconfounded, but it measures **local
calibration of the node marginal on structurally-related sets**, not branching ratios. Only the
edge/flux *interpretation* is invalidated by multi-parenthood — and that interpretation is what
justified the machinery.

Consequences:

- The defect is reachable inside the existing flat categorical.
- Consistent with Stage 3 (adding edge structure moved nothing) and with
  `project_capacity_vs_generalization` (bottleneck is generalization, not expressiveness → reshape
  what capacity fits rather than adding modules).

## 2. Coverage gate (measured, `scripts/eval/branch_beta_target_coverage.py --allow_multi_parent`)

1000 NIST20 train mols / 18,730 spectra.

The `indeg == 1` single-parent filter inherited from the edge-β design is **structurally fatal**:
only the ROOT has ≥2 single-parent children, because depth-1 fragments are the only nodes reachable
by one cut order. Measured 1000 such parents on 1000 molecules.

| | strict (single-parent) | relaxed (multi-parent OK) |
|---|---|---|
| eligible parent-spectra | 18,730 | 2,654,044 |
| alias-free | 620 (3.31%) | 671,382 (25.30%) |
| mean explained-intensity frac | 0.168 | 0.285 |
| strata with ≥2 NCE | 120 | 88,938 |

Dropping the filter is legitimate because the target is a marginal ranking, not edge flux.

**Margin sensitivity** (targets are not near-ties): 0.00 → 671,382 (flip 60.7%); **0.10 → 583,539**,
87% retained (flip 54.4%); 0.25 → 491,602, 73% retained (flip 45.9%). Coverage spans every NCE bin
and rises with energy (11.9% at 0–20 → ~27% at 50–120).

Caveats: flip rates are not comparable to the strict run's 33.3% (120 strata) — an "any set differs"
counter rises mechanically with more/larger strata. Sibling sets overlap (~3 parents/node), so the
effective independent sample is well below 583k, and alias-freeness is checked only *within* a
parent's child set.

## 3. Design

### Target builder (offline, once)

Per `(mol_id, spec_id)`: match peaks to nodes (`_peak_node_match`), then for each parent with ≥2
children keep the group if **alias-free** (no peak claimed by >1 observed sibling). Emit child node
indices and the normalized observed distribution `q_obs`.

On alias-free groups the min/max attribution bracket **collapses** — `unique == sub` in
`_sibling_dists`, so `w_min == w_max` identically. No bracket-robustness machinery is needed.

Output is ragged, keyed by `spec_id`: `group_ptr`, `child_nodes` (local DAG index), `q_obs`.
Sizing from the audit: ~31 groups/spectrum, 3–4 children each, ≈0.5 KB/spectrum.

### Loss — soft, not hard

Targets are **soft** (`q_obs`), not an argmax label. A hard dominant-child target would be a
per-candidate classification label derived from m/z-only matching, which
`feedback_candidate_level_hard_labels` rules out: alias-freeness removes the ambiguity *within* a
group, but a peak can still be claimed by nodes under other parents, so committing to a winner
asserts more than the data supports.

```python
lp   = pred_node_logprobs[sibrank_node_idxs]
logq = lp - scatter_logsumexp(lp, group_idxs)[group_idxs]      # = log q_base
loss = -(w_group * q_obs * logq).sum() / w_group.sum()          # KL(q_obs || q_base) + const
```

with `w_group` = the group's explained-intensity fraction (responsibility weighting).

Properties:

- **Supervises exactly `q_base`** — the same renormalization the eval performs, so loss and metric
  are the same object.
- **Constrains ratios only, not scale** — cannot fight the primary spectrum loss over how much total
  mass a sibling set receives, only over ordering inside it.
- **CE-conditioned without CE machinery** — labels are per spectrum, so the 54% dominant-child flips
  across NCE are taught through the targets rather than a new module. Given every CE-conditioned
  *module* to date has been null, this is the main structural bet.

Under the soft form the dominance margin stops defining the label and becomes an optional robustness
filter only; near-ties teach "these are comparable", which is true.

Cost: one gather plus one segment-logsumexp over ~30 groups/spectrum.

### Config

`config/template.yml`, next to the other loss weights, defaulting to off so existing configs are
bit-identical:

```yaml
sibling_rank_loss_weight: 0.0
sibling_rank_target_dp: null
```

### Arms

| arm | λ | purpose |
|---|---|---|
| baseline (`stage1_off`) | — | control |
| 0.01 / 0.1 / 1.0 | sweep | operating point |
| **shuffled labels** | best λ | confound control |

The shuffled arm keeps loss, group structure and gradient magnitude, permuting only the target
distribution. It is **not optional**: `project_capacity_vs_generalization` showed regularization is
worth ~+0.18 val and is unswept, so any aux term can raise val for reasons unrelated to branching.

### Stop-test

Val `dom_mis_model` falls from ~0.47 toward 0.11 **and** val cos does not regress **and** the
shuffled arm does not reproduce the cos change. `stage0_branching_model_vs_oracle.py` needs no
changes — it already computes `dom_mis_model` from `pred_node_probs`.

## 4. Known limitations

- **Reach is bounded.** Mean explained-intensity fraction 0.285 — sibling sets cover under a third of
  spectrum intensity. `dom_mis` can move substantially while cosine moves little. Decide in advance
  that this outcome is *informative* (it localizes the residual downstream of `P(n)`, where the
  fluxmpnn "downstream fork" already points), not a failure.
- **Overlapping groups.** ~3 parents/node means a node appears in several groups and receives several
  gradient contributions per spectrum; high-in-degree nodes are pulled harder. Optionally weight each
  entry by 1/(groups it appears in); left out of v1.
- **Selection bias, unmeasured.** The 25% of groups that survive alias-freeness are the m/z-resolvable
  ones, likely skewed toward heteroatom-differentiated cleavages and away from isomeric CH₂-shuffle
  cases. Cheap check: dump per-group descriptors and compare retained vs dropped composition.
- **Downstream interaction untested.** Reordering `P(n)` could perturb the `P(f|n)` composition; the
  cos-not-regressing half of the stop-test is what catches this.

## 5. Order of work

1. Target builder → run on full train split → report group count, on-disk size, retained-vs-dropped
   composition. **Decision point before any GPU time.**
2. Grade *existing* checkpoints with the targets (baseline, the built-but-untrained `frag_pool_combine=concat`
   arm, the unrun regularization sweep). The targets are a measurement instrument independent of the
   loss; if something already on the shelf moves `dom_mis`, that reorders priorities for one inference
   pass.
3. Implement the loss + config keys.
4. Run arms + shuffled control, apply the stop-test.
