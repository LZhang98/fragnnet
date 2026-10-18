# Fragment-connectivity branch `fc1`: local screen against `i1` is not positive

> **Status:** FINDING · **Updated:** 2026-09-29
> Step 3 of the user's three-step plan: a local screen on the RTX 2070 SUPER, with two seeds per
> arm on a quarter of the molecules. It is not comparable to Fir numbers. The read rule was fixed
> before any `fc1` result was seen.
>
> - **The branch.** Each candidate gets one message-passing layer over its own retained bonds,
>   from raw atom and bond features plus two per-atom counts: its retained degree and the cut
>   bonds it keeps. It is pooled over all atoms and over the cut-site atoms, and added to the
>   intensity token as a zero-initialized residual.
>   - It costs 21,504 parameters, +12% peak memory at batch 128, about 3% of step time, and 2-3
>     min per 1-hour run.
>   - Default off; `i1` reproduces bit for bit.
> - **The primary read passes neither bar.** The val tail (epochs 19-23) gains +0.0088 and
>   +0.0013 at the two seeds, a mean of +0.0050. The rule required a mean above the control pair's
>   own spread, 0.0076.
> - **Calibrated cosine erases most of it.** After cross-fitted `P^γ`, `fc1` is +0.0025 and
>   −0.0009 (mean +0.0008).
>   - Base-peak hit: +0.0057 and −0.0044.
>   - Ranking gap: −0.0037 and −0.0008.
> - **The decisions it targets do not move consistently.** At novelty < 0.4 nested balanced
>   accuracy is +0.002 and −0.018. Overlapping pairs are +0.031 and −0.009.
> - **It lowers train fit slightly**, like a weak regularizer: the eval-mode gap narrows by 0.012
>   and 0.004.
> - **Per the rule, the matched pooled control `fc1p` was not run.** This agrees with steps 1 and
>   2: describing the candidate after cutting does not recover the transfer deficit, at this
>   screen's resolution of about 0.01.

## Question

Does a small branch that describes each candidate after cutting help `i1` on validation
molecules? It covers the retained-fragment connectivity and the newly exposed cut sites, and it
supplements the pooled precursor atom embeddings. The prior was low:

- [step 1](masster_i1_base_peak_errors_2026_09_28.md) found a broad transfer deficit, not a
  missing distinction;
- [step 2](masster_i1_postcut_information_gate_2026_09_29.md) found hand-built post-cut
  descriptors worth at most about +0.001 cosine on held-out molecules;
- another session's
  retained-fragment plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_hierarchical_fragment_graph_plan.md`)
  deferred the branch on that evidence.

The user chose on 2026-09-29 to build and screen it anyway, as the direct test.

## The branch

Keys: `masster_fragment_connectivity_layers` (0 = off),
`masster_fragment_connectivity_hidden_dim` (32) and `masster_fragment_connectivity_messages`
(true). They are declared in `config/template.yml` and read by `MassterModel`
([`_fragment_connectivity_embeddings`](../../../src/fragnnet/model/masster_model.py)).

- **Graph.** Nodes are each candidate's (fragment, atom) membership rows. Edges are the precursor
  bonds with both endpoints in that fragment (`fragment_internal_edges` in
  [bond_fragment_heads.py](../../../src/fragnnet/model/bond_fragment_heads.py)). The same atom
  gets a different state in every candidate that cuts differently.
  - This differs from `r1_local_cut`, which was null: that arm re-sourced the cut endpoints from a
    k-hop view of the intact molecule.
- **Inputs.** Raw atom features (never the shared encoder's embeddings), plus the atom's retained
  degree in that fragment and the number of the fragment's cut bonds it keeps. Bond features come
  from `chemical_bond_attr`.
- **Encoder.** One plain GINE layer, width 32, with the model's precursor normalization and
  dropout.
- **Readout.**
  - Mean over all retained atoms, concatenated with the mean over cut-site atoms; the intact
    precursor gets zeros.
  - Projected to the token width and added after `input_projection`, before token dropout.
  - The projection is zero-initialized, so at step 0 the arm equals `i1` exactly (verified
    bit-identical in train and eval mode).
- **Matched control.** `messages: false` runs the same module, with the same parameters and
  features, on an empty edge set (`fc1p`).
- **Checks.**
  - 22 tests.
  - Against a brute force on real data (100,322 memberships), internal edges, degrees, counts and
    pools match exactly, including an atom in 136 fragments and 687 single-atom fragments.
  - With the new code, `i1` still reproduces Table 1 of step 1 exactly.
  - The 1,732 related MASSter tests pass. The full suite has one failure, in `test_runner_config`,
    from another session's in-progress relationship-loss code (`KeyError` on
    `masster_relationship_pairs_fp`), not this branch.
- **Cost.**
  - Parameters: 9,215,701 → 9,237,205.
  - At batch 128, peak memory 6.14 → 6.92 GB allocated (7.00 → 7.82 GB reserved).
  - Step time: the branch is about 13.8 ms of about 450 ms (join 3.2, GINE 7.3, pooling 3.3).
  - Wall time per 24-epoch run: 60-61 min, against 58 min for the uncontended control.
  - At batch 512 on Fir, the extrapolation is about +3 GB.
- **Arm configs:**
  - [`fc1_fragment_connectivity.yml`](../../../config/masster/mh_ft_memorization_route_2026_09_24/fc1_fragment_connectivity.yml);
  - [`fc1p_fragment_pooled.yml`](../../../config/masster/mh_ft_memorization_route_2026_09_24/fc1p_fragment_pooled.yml);
  - both `i1` plus the keys. Neither is in the Fir submitter.

## Setup

- **Recipe.** That of the [2026-09-27 local screens](../../history/masster_memorization_route_plan_log.md#local-screen-of-am1de1-2026-09-27):
  - a molecule-level subset of `nist20_inchikey`, 3,000 train and 1,000 val molecules;
  - `test_ids.csv` holds 1,000 train molecules, so the post-fit test pass is the eval-mode
    train fit;
  - batch 128, 24 epochs, on the relabelled `OB` / `OB_F` data.
- **Controls.** `i1_ctrl` (seed 397623) and `i1_ctrl_s2` (811237) were run fresh, because earlier
  local controls used `FT` labels. The arms are `fc1` and `fc1_s2` at the same seeds.
- **Probes.** Both base-peak probes ran on each run's best checkpoint, on the local val split and
  the train-molecule test split.
  - Novelty is taken against the 3,000 local train molecules, a mean nearest-neighbour Tanimoto of
    0.48.
  - Calibrated cosine is the tables' cross-fitted `P^γ`.

## Result

Training logs (val `cos_sim_0.01`; gap = best checkpoint's train-eval minus its val):

| run | tail 19-23 | tail 12-17 | best (epoch) | train-eval | gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `i1_ctrl` | 0.6586 | 0.6546 | 0.6675 (21) | 0.8914 | 0.2239 |
| `fc1` | 0.6674 | 0.6603 | 0.6705 (23) | 0.8829 | 0.2123 |
| `i1_ctrl_s2` | 0.6662 | 0.6592 | 0.6702 (23) | 0.8954 | 0.2253 |
| `fc1_s2` | 0.6675 | 0.6626 | 0.6708 (23) | 0.8920 | 0.2212 |

- Paired tail 19-23: **+0.0088 and +0.0013**, mean +0.0050.
- Tail 12-17: +0.0057 and +0.0034.
- Gap: −0.0116 and −0.0041.
- The control pair's own tail spread is 0.0076.

Probes on the best checkpoints (val unless marked):

| read | `i1_ctrl` | `fc1` | `i1_ctrl_s2` | `fc1_s2` |
| --- | ---: | ---: | ---: | ---: |
| candidate-space cosine | 0.6877 | 0.6910 | 0.6909 | 0.6921 |
| calibrated cosine (cross-fitted `P^γ`) | 0.6939 | **0.6964** | 0.6980 | **0.6971** |
| ranking gap | 0.1696 | 0.1659 | 0.1653 | 0.1645 |
| base-peak hit | 0.488 | 0.494 | 0.498 | 0.494 |
| hit, covered, novelty < 0.4 / 0.4-0.6 | 0.419 / 0.523 | 0.424 / 0.535 | 0.423 / 0.535 | 0.425 / 0.529 |
| nested, deeper winner / shallower winner | 0.549 / 0.798 | 0.559 / 0.791 | 0.576 / 0.802 | 0.535 / 0.814 |
| overlapping / disjoint pairs | 0.614 / 0.658 | 0.645 / 0.659 | 0.630 / 0.645 | 0.621 / 0.667 |
| nested balanced, novelty < 0.4 / 0.4-0.6 | 0.632 / 0.670 | 0.634 / 0.684 | 0.643 / 0.691 | 0.625 / 0.678 |
| train-molecule cosine | 0.9249 | 0.9162 | 0.9286 | 0.9247 |
| train-molecule ranking gap | 0.0261 | 0.0289 | 0.0240 | 0.0255 |

## Reading

- **By the rule fixed in advance, the screen is not positive.**
  - Both seeds gain on the tail, but the mean (+0.0050) is below the control pair's spread
    (0.0076).
  - Calibrated cosine, the metric the plan names, gains +0.0008 on average.
  - The raw gain is mostly the flattening that cross-fitted `P^γ` already gives.
- **The decisions the branch aims at do not move consistently.** Nested pairs at low novelty,
  the largest failing population in step 1, are +0.002 and −0.018. Overlapping pairs are +0.031
  and −0.009. No read moves beyond the seed pair's spread in the same direction at both seeds.
- **What it does do: fit training molecules a little less.**
  - Train-molecule cosine falls by 0.009 and 0.004, and the train ranking gap rises.
  - The eval-mode gap narrows by 0.012 and 0.004, with val flat to up. That is `am1`'s pattern at
    a smaller size, which fits a weak regularizer better than a new description of the fragment.
- **The three steps agree.** `i1`'s wrong base-peak choices are a transfer failure spread across
  chemistry (step 1). Explicit post-cut descriptors do not predict them (step 2). A learned
  post-cut encoding does not fix them at this resolution (step 3).
- **The matched pooled control `fc1p` was not run.** The rule runs it only after a positive read.
  Its config and key exist.

## Limits

- The screen resolves about 0.01. The tail difference between the two control seeds is 0.0076, so
  a true +0.003-0.005 effect is invisible here.
- Settling that needs `fc1` and `i1` at two seeds each on Fir (64 epochs, batch 512, about +3 GB
  per run). That is not recommended on this evidence, and not submitted.
- One design point: one layer, width 32, raw-feature inputs. Wider, deeper or
  embedding-seeded variants were not tried. The other plan treats deeper layers as a later
  hypothesis, not an automatic response to a null.
- The controls' best checkpoints are epochs 21 and 23, and both arms' are 23. The probe reads
  compare best checkpoints, as the plan's local screens do.

## Artifacts

`output/local_postcut_branch_2026_09_29/` (Git-ignored):
- `run_one.sh`, `run_controls.sh`, `run_arms.sh`, `run_probes.sh`, `parse_logs.py`,
  `prep_meta.py` / `meta_local.pkl`, `logs/`;
- `probe/`: per-run probe outputs and tables.

Checkpoints: `ckpts/local_{i1_ctrl,i1_ctrl_s2,fc1,fc1_s2}/`.

Tests: `tests/test_masster_fragment_connectivity.py` (22).
