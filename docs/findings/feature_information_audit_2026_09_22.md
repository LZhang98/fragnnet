# Feature and information audit, 2026-09-22 — what survived

> **Status:** FINDING — synthesis, read this before any of its five parts &nbsp;·&nbsp;
> **Updated:** 2026-09-22
>
> One investigation, started from "do ICEBERG and GLACIER have a larger model or more features?",
> ending at three proposed MASSter candidate-token changes. It produced five documents and
> **withdrew two of its own conclusions along the way**, so the map matters more than any single
> part. Nothing in it has been trained.

## Status of every claim

| claim | verdict | where |
|---|---|---|
| GLACIER is a bigger model | **Unevidenced.** It publishes no hidden dim, layer count or parameter count at all; only `4×A5000`. Its own baseline ICEBERG 2.0 is **7.0M** vs FraGNNet **2.79M**, MASSter **9.22M** (all measured) | [GLACIER note](../notes/glacier_object_detection.md) |
| They have more features | **Partly, and now closed.** ICEBERG has strictly fewer than us. GLACIER had exactly three we lacked and that are live on our data — added as `a_fused_ring`, `b_ring_size`, `b_fused_ring` | [GLACIER note](../notes/glacier_object_detection.md) |
| `b_stereo` would close the last gap | **No.** Preprocessing strips stereochemistry: 0 of 127,249 NIST20 bonds carry any, on NIST23 and MassSpecGym alike. A preprocessing decision, not a feature one | [feature headroom](feature_headroom_2026_09_22.md) |
| More descriptor features would help | **CLOSED NULL.** MMFF typing, EState, Crippen, TPSA/ASA and the benzylic/allylic/α-carbonyl family are all recoverable from current inputs at **0.93–1.00** held-out skill with two hops of context | [feature headroom](feature_headroom_2026_09_22.md) |
| Unused spectrum metadata is worth reading | **No.** `col_gas` is 99.5% N₂-or-missing, `ce_type` has no stepped spectra, the stepped-CE columns are 100% null. Only `res` is live, and it is a loss-tolerance question | [feature headroom](feature_headroom_2026_09_22.md) |
| A "correct" feature would displace a charge-retention shortcut | **NULL.** Size/mass alone reaches AUC 0.758 on a scaffold split and **degrades no faster** than composition (−0.025 vs −0.023 random→scaffold). Basicity class adds **−0.002** on the production split | [feature headroom](feature_headroom_2026_09_22.md#second-pass-shortcut-learning-2026-09-22) |
| There is a real shortcut in the feature set | **Yes, one.** `depth` proxies cut count on the plain BFS DAG and breaks on any multi-cut DAG. `num_cut_bonds` is the correct feature, is free, and is off in the prepass run where `depth` is wrong | [feature headroom](feature_headroom_2026_09_22.md) |
| We hold information we never use | **Yes.** NIST ships an expert formula per peak covering **99.4% of intensity** on 68.4% of spectra, read only by the GrAFF baseline; ClassyFire is materialized for 100% of molecules; `in_degree` has never run in production | [inventory](unused_information_inventory_2026_09_22.md) |
| MASSter needs a structural teacher | **RETRACTED.** Coverage already transfers (train `wrecall` 0.8975 vs val 0.8911 against a 0.151 cosine gap); GLACIER's "Not Converged" is a property of its DETR/LinSATNet proposal, not ours; same-formula isomers sum at one m/z so choosing between them changes the spectrum by nothing | [identity supervision](masster/masster_identity_supervision_2026_09_22.md) |
| MASSter's candidate token is missing cut information | **Open, measured.** **96.8%** of fragments are cut at ≥2 bonds and **55.4%** have two cuts sharing a ring, which mean pooling represents identically to unrelated cuts | [cut-set plan](../plans/masster/masster_cut_set_features_plan.md) |

## The one distinction that organises all of it

Three different questions get confused with each other, and they have different answers:

1. **Does the feature add information?** Almost never, for molecular descriptors — every
   2D-topological quantity is a deterministic function of a graph the encoder already sees. This
   is what the derivability probe settles.
2. **Is it the function the model actually fits?** A separate question, and the sharper one. A
   derivable feature can still lose to an easier non-causal correlate. Feature-availability
   experiments cannot answer it; only an intervention on a trained checkpoint can.
3. **Can the architecture still represent it at the point of use?** Not covered by either of the
   above. Information can reach the encoder, be computed, and then be **discarded or destroyed by
   pooling downstream** — at which point no upstream capacity restores it.

The descriptor null is question 1. The shortcut work is question 2. The live MASSter proposals
are question 3, which is why they are not closed by the null.

## What is in the working tree

**Code.** Three atom/bond features (`a_fused_ring`, `b_ring_size`, `b_fused_ring`) in the shared
registry with 10 tests; `masster_candidate_composition_features` (GLACIER eq. 14's
neutral-loss composition, off by default) with 4 tests and head plumbing. Full suite green, Ruff
clean on every changed file.

**Configs, none run.** `c4a_ring_topology` and `c4a_complement_composition`, one-key diffs from
`c4a_base_s1` at the same seed; FraGNNet `graph_feature_parity`.

**Diagnostics.** `debug_feature_headroom_probe.py`, `debug_charge_retention_shortcut.py`,
`debug_annotation_vs_dag_identity.py`, `debug_masster_identity_degeneracy.py`,
`debug_cut_set_topology.py`.

## What is proposed and not built

The three MASSter candidate-token changes, in the order they should be tried — remove
`use_base_probability_feature` (config only), add cut-set topology scalars, then feed the break
head's discarded per-bond states forward with a detached control. See the
[cut-set plan](../plans/masster/masster_cut_set_features_plan.md).

## Corrections made during this work

Recorded because they are the part most likely to mislead someone reading only one document.

1. **Derivability was the wrong sole criterion.** It answers "is this information", not "is this
   the function fitted". Flagged as a limitation in the original finding but under-weighted in its
   recommendation; the shortcut framing supersedes it.
2. **The MASSter teacher recommendation was wrong and is withdrawn.** It imported GLACIER's
   ablation across architectures without checking the mechanism it described, and it proposed
   fixing the candidate set — the one part that already transfers.
3. **NIST annotations were over-sold for MASSter specifically.** They are formula labels, and
   formula ambiguity is only 3.2% of MASSter's matched intensity. They remain the largest
   unused data in the repository and stay valuable for the DAG-based models and for rule mining.

## Open next checks

- **The corruption probe**, which is the instrument all three open questions need: randomize one
  feature group at a time on a trained checkpoint and measure the prediction shift on train versus
  held-out structures. Needs a production checkpoint, not present locally.
- **Re-measure the cut-count distribution on real MASSter candidate batches**, not the depth-3
  DAG, before quoting the 96.8% / 55.4% split.
- **Re-run the annotation-vs-DAG identity measurement at corpus scale**, stratified by collision
  energy and adduct; the 8.1% mismatch is molecule-set sensitive (a 40-molecule run gives 5.5%).
