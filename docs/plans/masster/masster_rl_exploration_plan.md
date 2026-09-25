# MASSter Exploration and Policy-Learning Plan

> **Status:** SUPERSEDED 2026-08-19 &nbsp;·&nbsp; **Updated:** 2026-08-19
>
> Historical analysis only; nothing below was implemented. The active successor is
> the support decision gate, Bernoulli-mixture probe, and canonical autoregressive
> design in [masster_plan.md](masster_plan.md) Section 6.1.2.

## Why this doc is retained

[masster_plan.md](masster_plan.md) §7.2 implemented `bounded_additive`
stochastic mask proposals and left open whether a sequential or RL policy should
follow. This doc records the staged options, the gate each one must pass, and
two mathematical constraints that rule out the cheap versions of the first two
stages.

The plan is superseded as an execution roadmap:

- R0's support/allocation question is absorbed by the main plan's oracle-coverage
  and minimum-explaining-cut diagnostics.
- R1 changes the objective to full-tail marginal estimation, which is not the
  current model objective.
- R2 and R3 may broaden training support but retain the independent inference
  scorer, so they do not remove the correlated-cut/downward-closure constraint.
- R4 duplicates the main plan's canonical autoregressive replacement; PPO is not a
  prerequisite for that normalized model.

The likelihood and estimator analysis below remains useful historical rationale.
Do not execute the staged order without reopening the plan against the newer main
plan decision gate.

**The premise is not yet established.** The Stage 2 note in the main plan
records that mixed-mask exploration improved candidate weighted recall but not
validation cosine. `candidate_wrecall` is computed on the *training* candidate
set (`src/fragnnet/pl_model/masster_pl.py`), which contains the exploration
masks, so a gain there is partly mechanical: adding masks to the set raises the
set's recall whether or not the bond model learned anything. Validation uses
deterministic exact top-K. Do not build a policy on top of this evidence.

## Terminology correction

The `masster_explore_masks` mechanism is **stochastic support
augmentation**. It broadens the candidate set that the spectrum loss can assign
probability to. It is not importance sampling, it does not estimate the full
mask marginal, and its retained masks carry no correction weight. Describe it
that way in configs and logs.

## R0 — Gate: support or allocation?

Run before committing to any policy work.

1. Log deterministic-path `candidate_wrecall` on validation next to the training
   value. The training number is not evidence about exploration.
2. **Rank migration.** For explored masks that contributed to an observed peak,
   record their deterministic top-K rank in later epochs. Exploration is only
   teaching the bond head if those masks migrate into the deterministic set.
   If they do not, exploration is padding the training candidate set and the
   gain cannot be realized at inference.
3. **Oracle-support ceiling.** Force-include the exploration masks at validation
   and measure cosine. This is the upper bound for any exploration method.

**Stop condition:** if the oracle-support ceiling is close to deterministic
cosine, the deficit is in charge retention, ion mode/H-transfer, or intensity
allocation, and no amount of mask exploration addresses it. Close this doc and
work the allocation path instead.

## Two constraints that shape everything below

### C1 — Self-normalized importance sampling is biased

An estimator cannot be both self-normalized and unbiased. If the requirement is
unbiasedness verified against brute-force enumeration on small `B`, the
estimator must not be self-normalized.

### C2 — The current proposal has no tractable `q(C)`

`bounded_additive` draws a cut count from a fixed prior, then samples that many
bonds with Gumbel top-k, then rejects masks already present in the deterministic
set, the forced set, or the accepted set. The resulting `q(C)` depends on the
cut-count mixture, the without-replacement Plackett-Luce set probability
(a sum over orderings), the deterministic exclusions, duplicate rejection, and
the masks accepted earlier for the same molecule. There is no per-mask
`log q(C)` to record, and no differentiable set likelihood either.

Kool et al. give without-replacement estimators and a follow-up unbiased
gradient estimator, but adopting them is a sampler redesign, not a metadata
field.

**Consequence:** C2 blocks both the importance-sampling stage and the
policy-gradient stage as originally sketched. Both need a proposal whose
likelihood is exactly computable — either an IID with-replacement Bernoulli
product, or a sequential per-bond factorization where the trajectory likelihood
is tractable by construction.

## R1 — Unbiased tail estimation *(only if the objective becomes full-marginal
estimation)*

Deterministic top-K contributions are summed exactly. A separate estimator
covers the remaining mask space:

```
P_tail(F) ~= (1/M) * sum_i  [ p(C_i) / q(C_i) ] * a(F, C_i) * 1{C_i not in D}
```

with `C_i ~ q` drawn **IID with replacement**, `D` the deterministic top-K set,
and `a(F, C)` the indicator that `F` is a component of `G \ C`.

- `q` is a fixed Bernoulli product, so
  `log q(C) = sum_b [ c_b log q_b + (1 - c_b) log(1 - q_b) ]` is exact.
- No self-normalization anywhere, per C1.
- Duplicates are **kept**. With-replacement sampling is what makes the estimator
  unbiased; the current rejection step is what breaks it.
- Keeping the indicator against the untruncated `q` avoids a complement
  normalizer. Renormalizing `q` over `D`'s complement is also exact, since `|D|`
  is a few hundred and `sum_{C in D} q(C)` is directly evaluable, but it buys
  nothing.
- With `q` detached, `grad_theta` of this estimator is unbiased for the gradient
  of the tail sum, so it can serve as a training objective and not only a
  diagnostic.

**Variance, not plumbing, decides viability.** At `q = p` the ratio is
identically one and the estimator degenerates to a tail hit count, which is the
low-variance choice for estimating tail *mass*. Tempering `q` to reach masks
that `p` ignores is exactly what reintroduces heavy-tailed ratios. Measure the
effective sample size before adopting.

**Cost that is easy to under-weight:** [masster_plan.md](masster_plan.md)
§3 currently *defines* `P_ion` as the truncated top-K marginal. Adding a tail
term changes the meaning of the model output, so existing MASSter numbers
become non-comparable. This is a modeling change requiring its own control run,
not a correctness patch.

## R2 — Learned proposal policy with REINFORCE

Separate the proposal `q_phi` from the scorer `p_theta`. Today they are the same
tempered logits, so exploration can only sample near what the model already
believes.

- `q_phi` is a small head on the existing bond embeddings, replacing the fixed
  `masster_exploration_cut_count_weights` prior.
- **Prerequisite (C2):** the proposal must be IID with replacement, or a
  sequential per-bond policy. `grad_phi log q_phi(C)` does not exist for the
  current sampler.
- **Reward must be responsibility-based, not hard m/z matching.**
  `r(C) = sum over pathways of C of resp(pathway) * true_intensity`, where
  `resp` is the pathway's posterior share of an observed peak under the existing
  logsumexp aggregation. m/z matching is many-to-one; hard credit rewards
  chemically wrong assignments that happen to land on-mass.
- **Baseline:** RLOO over the masks sampled for the same molecule. Many samples
  per precursor already exist, so this is nearly free and lower-variance than a
  learned critic. Do not build a value network.
- Keep the RL term out of `theta`. Turning the calibrated Bernoulli scorer into
  a reward maximizer destroys the exact top-K semantics in §2.1 of the main plan.
- Exposing per-pathway responsibility requires the sparse CE to return it;
  today it returns aggregates only.

**Gate:** rank migration (R0.2) must improve. If `q_phi` finds peak-explaining
masks that `p_theta` still will not rank, the bond scorer is the limit and this
is the wrong lever.

## R3 — GFlowNet over cut masks

If R2 shows migration but plateaus. Reward-maximizing RL mode-collapses, and
mode collapse is the one failure a candidate generator cannot absorb.

- State is a partial cut set, action adds one bond or stops, terminal state is a
  cut mask; trajectory-balance loss with `R(C)` from R2.
- Sampling proportional to reward gives diversity without entropy tuning.
- Multiple orderings reach the same mask. Use a canonical order (sorted bond
  index) to make trajectories unique, or handle the full order lattice with
  proper flow matching.
- Can replace the proposal while leaving `p_theta` as the scorer, so exact top-K
  at inference survives.

## R4 — Autoregressive cut policy *(listed, not started)*

The only path that addresses main-plan §9.3, correlated/concerted cleavage:
condition bond `k` on the bonds already broken, trained with PPO against the
spectrum reward.

- Breaks the cheap exact top-K heap. The §2.1 proof depends on additive flip
  penalties, so practical enumeration requires approximate beam or best-first
  search. A canonical policy normalized over the next admissible bond plus `STOP`
  still assigns an exact probability to each emitted cut set, preserving normalized
  retained/OOS mass even though search recall is approximate.
- ICEBERG precedent exists, but its autoregressive atom removal is the mechanism
  MASSter was designed to avoid.
- Do not open until R2/R3 show that mask support, not scoring, limits cosine.

## Cross-cutting requirements

| Item | Requirement |
|------|-------------|
| Reward | Responsibility-weighted; never hard peak matching |
| Baseline | RLOO over in-molecule samples; no critic network |
| Evaluation | Deterministic exact top-K always; policies are training-only |
| Seeds | At least 3, per main plan §7.2; exploration variance is large |
| Budget matching | Compare at matched *realized* mask and component counts |
| Cost | Selection is already the CPU bottleneck (main plan §7.2.1); a learned proposal adds a device round trip per step |
| Reporting | Stratify by cut count — policies preferentially find high-cut masks, which §9.2 already flags as mis-calibrated |

## Historical recommended order — do not execute

1. R0 rank migration, plus the terminology fix.
2. R2 only if R0 shows migration, built on a tractable proposal from the start.
3. R1 only if the objective actually becomes full-marginal estimation, with its
   own control run.
4. R3, then R4, each gated on the previous result.

## Related

- [masster_plan.md](masster_plan.md) — §2.1 exact top-K, §3 truncated
  marginal, §6.1.2 active correlated-cut decision path, §7.2 support augmentation,
  and §9 open risks.
- `src/fragnnet/frag/masster_candidates.py` — proposal and rejection loop.
- `src/fragnnet/pl_model/masster_pl.py` — `candidate_wrecall` logging.
- `config/masster/merged/ablations/bond_model/nist20_merged_mask256_gine_gsa384_bondcentric_a2_b3.yml`
  and its `_explore64` / `_explore256` variants — the deterministic control and
  the augmented arms.
