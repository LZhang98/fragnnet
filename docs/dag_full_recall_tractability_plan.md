# DAG Full-Recall Tractability Plan

## Motivation

`docs/bond_breakability_mask_plan.md` and `docs/dag_fanout_cap_plan.md` both
mined statistical policies for pruning the D4 DAG (bond-type masking,
per-parent fan-out capping) to control node count. Both are explicitly lossy:
their best policies trade 0.74%-8.79% recall loss for node/edge reduction, and
a learned scorer discussed as a follow-on (rank candidates instead of a hand-
mined bucket table) would still be lossy in the same way — a statistical
ranking model cannot guarantee zero false negatives on unseen molecules
without fully enumerating them anyway, which defeats the purpose.

**Requirement clarified 2026-07-03: 100% DAG structural recall.** Every true
fragment/formula reachable by full enumeration must be reachable in the
generated DAG — no bond masking, no fan-out capping, no learned pruning of any
kind, regardless of how small the measured loss is on a mining sample. This
disqualifies the "learned bond-priority scorer" direction outright: it is not
a smaller/safer version of the mask/cap plans, it is the same category of
lossy approximation and is incompatible with this requirement.

Full unmasked D4 already gives 100% recall (the "unlimited" row in both prior
docs' tables). The problem this doc addresses is therefore **tractability**,
not recall: a small number of molecules blow up combinatorially under full
enumeration (`bond_breakability_mask_plan.md` Step 1: 33/3,000 sampled
molecules exceeded 50,000 DAG nodes at D4; the worst case was 1,346,800 nodes
/ 9,720,683 edges for a single molecule). Since nodeMLP cost — the dominant
downstream cost driver, established in `dag_fanout_cap_plan.md` — scales with
node count, and node count cannot be reduced without losing recall, the fix
has to move to a different stage of the pipeline rather than the DAG
structure itself.

## Two workstreams

These are independent and can be sequenced or split:

### Workstream A — Generation-time engineering for tractable full enumeration

Make `multi_cut_bfs.pyx` handle the outlier molecules without OOM/blowing up
preprocessing time, without changing what gets enumerated:

- Profile where cost actually goes for the 1.3M-node outlier case (dedup
  lookup, candidate generation, bitmask ops, Python-level overhead vs. Cython
  loop cost).
- Better dedup data structures / memory layout for `css_to_id_dict` at scale.
- Streaming or on-disk fragment storage for outlier molecules instead of
  all-in-memory, so generation doesn't require holding a 1.3M-node DAG in RAM
  at once.
- Quarantine the ~1% pathological molecules into a separate slow-path batch
  with a larger time/memory budget during corpus preprocessing, rather than
  letting them dictate a global policy (mask/cap) that costs the other ~99%
  of molecules recall for no benefit to their own tractability.

→ verify: regenerate the known 1.3M-node outlier and confirm it completes
within an acceptable wall-clock/memory budget, with identical node/edge
counts to the current unmasked baseline (no change in what's enumerated).

### Workstream B — Decouple training-time compute from DAG completeness

The DAG only needs to be exhaustively generated once, offline, per molecule.
Recall is a property of what's *in* the DAG, not of which nodes receive a
forward/backward pass on any given training step. So:

- Generate the full DAG (100% recall, can be slow/expensive since it is
  one-time preprocessing per molecule).
- At training time, subsample nodes per batch for the nodeMLP forward/backward
  pass — ordinary minibatch-style stochastic subsampling. This is an unbiased
  compute/variance tradeoff, not a recall loss, since eval/inference can still
  walk the full DAG.
- Optionally, use importance-weighted (rather than uniform) sampling to
  reduce gradient variance — this is the one place a learned/heuristic
  component could reenter legitimately, but strictly as a training-speed
  optimization. It must never cause a node to be dropped from the DAG itself
  or from eval-time traversal.
- Inference/eval-time cost for pathological molecules remains a separate,
  smaller concern (bounded to prediction time, not the tight training loop),
  and can afford a larger time budget than a training minibatch.

→ verify: training-time node subsampling produces loss/gradient estimates
consistent with full-node passes on small/medium DAGs (where full passes are
still affordable, as a sanity baseline), and confirm eval-time recall metrics
are computed over the complete DAG, not the subsampled set.

## Explicitly disqualified

- Bond-type masking (`bond_breakability_mask_plan.md`) — lossy by
  construction, incompatible with the 100% recall requirement.
- Per-parent fan-out capping (`dag_fanout_cap_plan.md`) — same issue, at the
  node-creation level instead of the bond level.
- Learned bond-priority scorer as a replacement ranking function for either
  of the above — inherits the same incompatibility; a statistical model
  cannot certify zero false negatives on unseen molecules.

## Open decision

Scope order not yet chosen: (A) generation-time engineering first, (B)
training-time decoupling first, or both in parallel. They address different
failure modes (A = preprocessing can't finish / OOMs; B = training is too
slow once the DAG exists) and can proceed independently.

## Related docs

- `docs/bond_breakability_mask_plan.md` — mined bond-level masking; superseded
  as a candidate mechanism by the 100% recall requirement, but its Step 1
  numbers (outlier molecule counts, node/edge counts) are the source of truth
  for how bad the tractability problem is.
- `docs/dag_fanout_cap_plan.md` — mined per-parent fan-out capping; same
  disqualification, same reason.

## Key files

- `src/fragnnet/frag/multi_cut_bfs.pyx` — Workstream A target (dedup,
  candidate generation, per-molecule memory footprint).
- `src/fragnnet/model/nn_blocks.py:620-675` (nodeMLP) and
  `src/fragnnet/model/fragnnet_model.py:~1345` (invocation) — Workstream B
  target for training-time node subsampling.
- `preproc_scripts/03_prepare_dag_feats.py` — where a quarantine/slow-path
  batch mechanism for outlier molecules would be added (Workstream A).
