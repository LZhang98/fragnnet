# MASShatter attention rework + parameter budget — implementation record

> **Status:** BUILT, NOT RUN &nbsp;·&nbsp; **Updated:** 2026-09-21
>
> Phase 2 (attention) and Phase 3 (bond head / budget) of the 2026-09-21 todo list are
> implemented and tested. **Every option defaults to the historical behaviour**, so `main`
> behaviour is unchanged until an arm config is launched. No training has been run.

## Why these, and not more coverage

The measured position: `val_candidate_wrecall` is 0.895 and buying more support *cost*
cosine (enum1024→2048 raised fill 0.699→0.867 while best cosine fell 0.7373→0.7340). The
gap is allocation among candidates that are already on the ballot. Re-measuring the
parameter budget on 2026-09-21 showed the model now spends **41.9% of 9.22M on the bond
head** (candidate generation, saturated) and **26.7% on the intensity head** (allocation,
the bottleneck) — and the module that actually emits intensities is ~2k parameters, so the
candidate attention stack *is* the allocator.

## What landed

### Phase 2 — candidate attention

| option | key | default |
|---|---|---|
| Normalization placement | `mass_shatter_intensity_norm_placement` | `post` |
| Query/key LayerNorm | `mass_shatter_intensity_qk_norm` | `False` |
| Neutral-loss pair bias | `mass_shatter_intensity_mass_pair_bias` | `False` |

- **`_BiasedSelfAttention`** (`model/bond_fragment_heads.py`) replaces `nn.MultiheadAttention`
  inside `_FullCandidateAttentionLayer`. Parameter names mirror the stock module
  (`in_proj_weight`, `in_proj_bias`, `out_proj`), and the outputs are **bit-identical**
  (max abs diff 0.0, pinned by a test), so this is a true drop-in and existing
  full-attention checkpoints load unchanged. It exists because neither a pairwise additive
  bias nor QK-norm can be expressed through `nn.MultiheadAttention` without materializing
  an `(molecules·heads, candidates, candidates)` mask.
- **`PairwiseMassBias`** adds `b_ij = Σ_r w_r·RBF(m_i − m_j; μ_r, σ_r)` to the attention
  logits, with centres initialized on common neutral losses (H₂, NH₃, H₂O, CO, C₂H₄, CO₂,
  HCOOH) and mirrored for both signs; centres, widths and weights are all learnable.
  **Weights initialize to zero**, so enabling the bias is an exact no-op at step 0 and is
  safe on a warm start (pinned by a test). The bias is shared across heads: a per-head
  variant costs one head-multiple more memory than the attention itself.
- **Pre-norm** normalizes the branch input rather than the residual sum, in both the induced
  and full layers. Parameter names and counts are identical across placements, so post/pre
  is A/B-comparable from one checkpoint.
- Pair bias and QK-norm **require `attention_type: full`** and raise otherwise — the induced
  layer routes information through latents that carry no mass of their own.

### Phase 3 — bond head and budget

| option | key | default |
|---|---|---|
| Incident-bond pooling | `mass_shatter_bond_endpoint_aggregation` | `mean` |
| GNN residual branch activation | `gnn_residual_activation` | `relu` |
| Weight-decay parameter groups | `weight_decay_exclude_norm_bias` | `False` |

- **`endpoint_aggregation`** on `BondFocusedBreakHead` adds `sum` and `sqrt` beside the
  historical `mean`. A mean cannot separate a quaternary carbon from a methyl, which is
  GIN's own expressiveness argument, and this head is the largest module in the model.
- **`gnn_residual_activation`** covers all six residual sites in `nn_blocks.py`
  (GINE, GINE_GSA, GatedGINE, GAT, MPNN, NodeMLP). `relu` means a layer can only *add*
  per-coordinate mass to the residual stream and can never attenuate it; `identity` removes
  that restriction. A test pins the behavioural difference (branch minimum ≥ 0 vs < 0).
- **`weight_decay_exclude_norm_bias`** splits the optimizer into two groups so LayerNorm and
  BatchNorm weights, every bias, and 1-D learned tensors (including the candidate
  transformer's `inducing_tokens`, initialized at `std=model_dim**-0.5` and then pulled
  toward zero every step) are not decayed.

## Arms to run

`config/massshatter/ma_mi_merged/attention_2026_09_21/`, all derived from
`nist20_ma_mi_merged_fast_batch512_cuda_lb64.yml` (base **9.217M**):

| arm | change | params |
|---|---|---:|
| `a1_prenorm` | pre-norm | 9.217M |
| `a2_full_attention` | full candidate attention | 8.147M |
| `a3_full_massbias` | full + neutral-loss pair bias | 8.147M |
| `a4_full_massbias_prenorm_qknorm` | all four attention changes | 8.147M |
| `b1_bond_sum` | sum-pooled incident bonds | 9.217M |
| `b2_identity_residual` | ungated GNN residual | 9.217M |
| `b3_wd_groups` | weight-decay groups | 9.217M |
| `b4_budget_rebalance` | bond head 3→1 layers, transformer 256→384 wide, 128 inducing | **9.185M** |

`b4` is deliberately **parameter-neutral** (−0.3%): it tests where capacity should sit, not
how much there is. Run `a2` before `a3`/`a4` so a pair-bias win can be attributed, and
screen at ~128 epochs — val cosine saturates near ep100.

## Not implemented here

- **Atom-overlap / containment bias** (Jaccard on candidate atom masks). Needs ragged
  atom-membership densified into the head's per-bucket padded layout; the mass bias needs
  only a per-candidate scalar, so it landed first.
- **Same-ring-system bias.** Depends on `frag/ring_system_cuts.py` plumbing that is built
  but not wired into the model.
- **CE-conditioned attention.** Explicitly gated on the Phase 0 measurements, since
  `[[project_ce_extent_allocation_null]]` and `[[project_stage0_branching_headroom_model]]`
  both concluded the error is CE-invariant.

## Verification

3,824 tests pass (`-k "not cuda and not benchmark and not slow"`). 18 new tests in
`tests/test_masshatter_attention_and_budget.py` cover drop-in equivalence, the zero-init
no-op property, padding safety, name-stability across norm placements, the guardrails, the
aggregation modes, the residual-branch behaviour, and the weight-decay split. `ruff check`
reports the same 138 pre-existing errors before and after. Every arm config was built
through `MassShatterPL(**cfg)` and its parameter count recorded above.
